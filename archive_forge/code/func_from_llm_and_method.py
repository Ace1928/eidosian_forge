from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool
from langchain_community.tools.openapi.utils.api_models import APIOperation
from langchain_community.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain_community.utilities.requests import Requests
@classmethod
def from_llm_and_method(cls, llm: BaseLanguageModel, path: str, method: str, spec: OpenAPISpec, requests: Optional[Requests]=None, verbose: bool=False, return_intermediate_steps: bool=False, **kwargs: Any) -> 'NLATool':
    """Instantiate the tool from the specified path and method."""
    api_operation = APIOperation.from_openapi_spec(spec, path, method)
    chain = OpenAPIEndpointChain.from_api_operation(api_operation, llm, requests=requests, verbose=verbose, return_intermediate_steps=return_intermediate_steps, **kwargs)
    return cls.from_open_api_endpoint_chain(chain, spec.info.title)