from __future__ import annotations
import json
from typing import Any, Dict, List, NamedTuple, Optional, cast
from langchain_community.tools.openapi.utils.api_models import APIOperation
from langchain_community.utilities.requests import Requests
from langchain_core.callbacks import CallbackManagerForChainRun, Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field
from requests import Response
from langchain.chains.api.openapi.requests_chain import APIRequesterChain
from langchain.chains.api.openapi.response_chain import APIResponderChain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
@classmethod
def from_url_and_method(cls, spec_url: str, path: str, method: str, llm: BaseLanguageModel, requests: Optional[Requests]=None, return_intermediate_steps: bool=False, **kwargs: Any) -> 'OpenAPIEndpointChain':
    """Create an OpenAPIEndpoint from a spec at the specified url."""
    operation = APIOperation.from_openapi_url(spec_url, path, method)
    return cls.from_api_operation(operation, requests=requests, llm=llm, return_intermediate_steps=return_intermediate_steps, **kwargs)