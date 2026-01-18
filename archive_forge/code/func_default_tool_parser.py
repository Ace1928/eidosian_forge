import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict
from langchain_core.messages.base import (
from langchain_core.utils._merge import merge_dicts
def default_tool_parser(raw_tool_calls: List[dict]) -> Tuple[List[ToolCall], List[InvalidToolCall]]:
    """Best-effort parsing of tools."""
    tool_calls = []
    invalid_tool_calls = []
    for tool_call in raw_tool_calls:
        if 'function' not in tool_call:
            continue
        else:
            function_name = tool_call['function']['name']
            try:
                function_args = json.loads(tool_call['function']['arguments'])
                parsed = ToolCall(name=function_name or '', args=function_args or {}, id=tool_call.get('id'))
                tool_calls.append(parsed)
            except json.JSONDecodeError:
                invalid_tool_calls.append(InvalidToolCall(name=function_name, args=tool_call['function']['arguments'], id=tool_call.get('id'), error='Malformed args.'))
    return (tool_calls, invalid_tool_calls)