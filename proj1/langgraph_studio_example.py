from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] ="My First App"

from langchain.chat_models import init_chat_model
llm = init_chat_model(model = "groq:qwen/qwen3-32b")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_tool_graph():
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    tools = [add]
    tool_node= ToolNode([add])


    llm_with_tools = llm.bind_tools(tools=tools)

    # StateGraph creation

    # node definition
    def tool_calling_llm(state:State):
        return {
            'messages': [llm_with_tools.invoke(state['messages'])]
        }

    # Graph
    builder = StateGraph(State)
    builder.add_node('tool_calling_llm', tool_calling_llm)
    builder.add_node('tools', ToolNode(tools))

    # add Edges
    builder.add_edge(START, 'tool_calling_llm')
    builder.add_conditional_edges(
        'tool_calling_llm', tools_condition
    )
    builder.add_edge('tools', 'tool_calling_llm')

    graph = builder.compile()
    return graph


agent = make_tool_graph()

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like in San Francisco?"}]
})
print(result['messages'][-1].content)