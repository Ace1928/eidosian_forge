from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool
from langchain_community.tools.openapi.utils.api_models import APIOperation
from langchain_community.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain_community.utilities.requests import Requests
@classmethod
def from_open_api_endpoint_chain(cls, chain: OpenAPIEndpointChain, api_title: str) -> 'NLATool':
    """Convert an endpoint chain to an API endpoint tool."""
    expanded_name = f'{api_title.replace(' ', '_')}.{chain.api_operation.operation_id}'
    description = f"I'm an AI from {api_title}. Instruct what you want, and I'll assist via an API with description: {chain.api_operation.description}"
    return cls(name=expanded_name, func=chain.run, description=description)