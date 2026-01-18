from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@property
def body_params(self) -> List[str]:
    if self.request_body is None:
        return []
    return [prop.name for prop in self.request_body.properties]