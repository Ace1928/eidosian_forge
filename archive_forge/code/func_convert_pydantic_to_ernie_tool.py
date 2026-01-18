from typing import Literal, Optional, Type, TypedDict
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
def convert_pydantic_to_ernie_tool(model: Type[BaseModel], *, name: Optional[str]=None, description: Optional[str]=None) -> ToolDescription:
    """Convert a Pydantic model to a function description for the Ernie API."""
    function = convert_pydantic_to_ernie_function(model, name=name, description=description)
    return {'type': 'function', 'function': function}