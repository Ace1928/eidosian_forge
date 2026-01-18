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
def format_tool_to_openai_function(tool: BaseTool) -> FunctionDescription:
    """Format tool into the OpenAI function API."""
    if tool.args_schema:
        return convert_pydantic_to_openai_function(tool.args_schema, name=tool.name, description=tool.description)
    else:
        return {'name': tool.name, 'description': tool.description, 'parameters': {'properties': {'__arg1': {'title': '__arg1', 'type': 'string'}}, 'required': ['__arg1'], 'type': 'object'}}