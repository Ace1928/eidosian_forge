from __future__ import annotations
import importlib
from typing import (
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import Literal
def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get('role')
    if role == 'user':
        return HumanMessage(content=_dict.get('content', ''))
    elif role == 'assistant':
        content = _dict.get('content', '') or ''
        additional_kwargs: Dict = {}
        if (function_call := _dict.get('function_call')):
            additional_kwargs['function_call'] = dict(function_call)
        if (tool_calls := _dict.get('tool_calls')):
            additional_kwargs['tool_calls'] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == 'system':
        return SystemMessage(content=_dict.get('content', ''))
    elif role == 'function':
        return FunctionMessage(content=_dict.get('content', ''), name=_dict.get('name'))
    elif role == 'tool':
        additional_kwargs = {}
        if 'name' in _dict:
            additional_kwargs['name'] = _dict['name']
        return ToolMessage(content=_dict.get('content', ''), tool_call_id=_dict.get('tool_call_id'), additional_kwargs=additional_kwargs)
    else:
        return ChatMessage(content=_dict.get('content', ''), role=role)