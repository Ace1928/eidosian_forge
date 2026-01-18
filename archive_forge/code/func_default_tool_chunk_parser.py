import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict
from langchain_core.messages.base import (
from langchain_core.utils._merge import merge_dicts
def default_tool_chunk_parser(raw_tool_calls: List[dict]) -> List[ToolCallChunk]:
    """Best-effort parsing of tool chunks."""
    tool_call_chunks = []
    for tool_call in raw_tool_calls:
        if 'function' not in tool_call:
            function_args = None
            function_name = None
        else:
            function_args = tool_call['function']['arguments']
            function_name = tool_call['function']['name']
        parsed = ToolCallChunk(name=function_name, args=function_args, id=tool_call.get('id'), index=tool_call.get('index'))
        tool_call_chunks.append(parsed)
    return tool_call_chunks