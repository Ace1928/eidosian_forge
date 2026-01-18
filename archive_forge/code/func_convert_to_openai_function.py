from __future__ import annotations
import inspect
import uuid
from typing import (
from typing_extensions import TypedDict
from langchain_core._api import deprecated
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
def convert_to_openai_function(function: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI function.

    Args:
        function: Either a dictionary, a pydantic.BaseModel class, or a Python function.
            If a dictionary is passed in, it is assumed to already be a valid OpenAI
            function or a JSON schema with top-level 'title' and 'description' keys
            specified.

    Returns:
        A dict version of the passed in function which is compatible with the
            OpenAI function-calling API.
    """
    from langchain_core.tools import BaseTool
    if isinstance(function, dict) and all((k in function for k in ('name', 'description', 'parameters'))):
        return function
    elif isinstance(function, dict) and all((k in function for k in ('title', 'description', 'properties'))):
        function = function.copy()
        return {'name': function.pop('title'), 'description': function.pop('description'), 'parameters': function}
    elif isinstance(function, type) and issubclass(function, BaseModel):
        return cast(Dict, convert_pydantic_to_openai_function(function))
    elif isinstance(function, BaseTool):
        return cast(Dict, format_tool_to_openai_function(function))
    elif callable(function):
        return convert_python_function_to_openai_function(function)
    else:
        raise ValueError(f"Unsupported function\n\n{function}\n\nFunctions must be passed in as Dict, pydantic.BaseModel, or Callable. If they're a dict they must either be in OpenAI function format or valid JSON schema with top-level 'title' and 'description' keys.")