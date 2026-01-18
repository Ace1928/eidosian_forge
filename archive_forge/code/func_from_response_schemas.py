from __future__ import annotations
from typing import Any, List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.pydantic_v1 import BaseModel
from langchain.output_parsers.format_instructions import (
@classmethod
def from_response_schemas(cls, response_schemas: List[ResponseSchema]) -> StructuredOutputParser:
    return cls(response_schemas=response_schemas)