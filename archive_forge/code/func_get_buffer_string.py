from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from langchain_core.messages.ai import (
from langchain_core.messages.base import (
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
def get_buffer_string(messages: Sequence[BaseMessage], human_prefix: str='Human', ai_prefix: str='AI') -> str:
    """Convert a sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
        ai_prefix: THe prefix to prepend to contents of AIMessages.

    Returns:
        A single string concatenation of all input messages.

    Example:
        .. code-block:: python

            from langchain_core import AIMessage, HumanMessage

            messages = [
                HumanMessage(content="Hi, how are you?"),
                AIMessage(content="Good, how are you?"),
            ]
            get_buffer_string(messages)
            # -> "Human: Hi, how are you?
AI: Good, how are you?"
    """
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = 'System'
        elif isinstance(m, FunctionMessage):
            role = 'Function'
        elif isinstance(m, ToolMessage):
            role = 'Tool'
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f'Got unsupported message type: {m}')
        message = f'{role}: {m.content}'
        if isinstance(m, AIMessage) and 'function_call' in m.additional_kwargs:
            message += f'{m.additional_kwargs['function_call']}'
        string_messages.append(message)
    return '\n'.join(string_messages)