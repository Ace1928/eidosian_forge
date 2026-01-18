from typing import Any, Dict, List, Literal
from langchain_core.messages.base import (
from langchain_core.messages.tool import (
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.json import (
@root_validator()
def init_tool_calls(cls, values: dict) -> dict:
    if not values['tool_call_chunks']:
        values['tool_calls'] = []
        values['invalid_tool_calls'] = []
        return values
    tool_calls = []
    invalid_tool_calls = []
    for chunk in values['tool_call_chunks']:
        try:
            args_ = parse_partial_json(chunk['args'])
            if isinstance(args_, dict):
                tool_calls.append(ToolCall(name=chunk['name'] or '', args=args_, id=chunk['id']))
            else:
                raise ValueError('Malformed args.')
        except Exception:
            invalid_tool_calls.append(InvalidToolCall(name=chunk['name'], args=chunk['args'], id=chunk['id'], error='Malformed args.'))
    values['tool_calls'] = tool_calls
    values['invalid_tool_calls'] = invalid_tool_calls
    return values