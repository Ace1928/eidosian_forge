from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@classmethod
def from_request_body(cls, request_body: RequestBody, spec: OpenAPISpec) -> 'APIRequestBody':
    """Instantiate from an OpenAPI RequestBody."""
    properties = []
    for media_type, media_type_obj in request_body.content.items():
        if media_type not in _SUPPORTED_MEDIA_TYPES:
            continue
        api_request_body_properties = cls._process_supported_media_type(media_type_obj, spec)
        properties.extend(api_request_body_properties)
    return cls(description=request_body.description, properties=properties, media_type=media_type)