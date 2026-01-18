from __future__ import annotations
import inspect
import uuid
from typing import (
from typing_extensions import TypedDict
from langchain_core._api import deprecated
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
@deprecated('0.1.16', alternative='langchain_core.utils.function_calling.convert_to_openai_function()', removal='0.2.0')
def convert_python_function_to_openai_function(function: Callable) -> Dict[str, Any]:
    """Convert a Python function to an OpenAI function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
        the docstring has Google Python style argument descriptions, these will be
        included as well.
    """
    description, arg_descriptions = _parse_python_function_docstring(function)
    return {'name': _get_python_function_name(function), 'description': description, 'parameters': {'type': 'object', 'properties': _get_python_function_arguments(function, arg_descriptions), 'required': _get_python_function_required_args(function)}}