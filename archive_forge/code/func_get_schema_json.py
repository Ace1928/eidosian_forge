import asyncio
from functools import partial
from typing import Any, Dict, List, Optional, Type
from langchain_core.callbacks.manager import (
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from langchain_core.tools import BaseTool
from langchain_community.tools.connery.models import Action, Parameter
def get_schema_json(self) -> str:
    """
        Returns the JSON representation of the Connery Action Tool schema.
        This is useful for debugging.
        Returns:
            str: The JSON representation of the Connery Action Tool schema.
        """
    return self.args_schema.schema_json(indent=2)