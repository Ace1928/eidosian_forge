from typing import Any, List
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
def get_operations(self) -> List[dict]:
    """Return a list of operations."""
    return self.operations