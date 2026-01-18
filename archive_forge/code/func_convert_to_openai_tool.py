from __future__ import annotations
import inspect
import uuid
from typing import (
from typing_extensions import TypedDict
from langchain_core._api import deprecated
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
def convert_to_openai_tool(tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI tool.

    Args:
        tool: Either a dictionary, a pydantic.BaseModel class, Python function, or
            BaseTool. If a dictionary is passed in, it is assumed to already be a valid
            OpenAI tool, OpenAI function, or a JSON schema with top-level 'title' and
            'description' keys specified.

    Returns:
        A dict version of the passed in tool which is compatible with the
            OpenAI tool-calling API.
    """
    if isinstance(tool, dict) and tool.get('type') == 'function' and ('function' in tool):
        return tool
    function = convert_to_openai_function(tool)
    return {'type': 'function', 'function': function}