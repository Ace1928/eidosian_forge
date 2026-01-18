import json
from typing import List, Sequence, Tuple
from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage
def format_to_openai_function_messages(intermediate_steps: Sequence[Tuple[AgentAction, str]]) -> List[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations

    Returns:
        list of messages to send to the LLM for the next prediction

    """
    messages = []
    for agent_action, observation in intermediate_steps:
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))
    return messages