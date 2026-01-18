import json
from typing import List, Sequence, Tuple
from langchain_core.agents import AgentAction
from langchain_core.messages import (
from langchain.agents.output_parsers.tools import ToolAgentAction
def format_to_tool_messages(intermediate_steps: Sequence[Tuple[AgentAction, str]]) -> List[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations

    Returns:
        list of messages to send to the LLM for the next prediction

    """
    messages = []
    for agent_action, observation in intermediate_steps:
        if isinstance(agent_action, ToolAgentAction):
            new_messages = list(agent_action.message_log) + [_create_tool_message(agent_action, observation)]
            messages.extend([new for new in new_messages if new not in messages])
        else:
            messages.append(AIMessage(content=agent_action.log))
    return messages