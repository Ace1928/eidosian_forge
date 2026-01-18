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
def from_api_operation(cls, operation: APIOperation, llm: BaseLanguageModel, requests: Optional[Requests]=None, verbose: bool=False, return_intermediate_steps: bool=False, raw_response: bool=False, callbacks: Callbacks=None, **kwargs: Any) -> 'OpenAPIEndpointChain':
    """Create an OpenAPIEndpointChain from an operation and a spec."""
    param_mapping = _ParamMapping(query_params=operation.query_params, body_params=operation.body_params, path_params=operation.path_params)
    requests_chain = APIRequesterChain.from_llm_and_typescript(llm, typescript_definition=operation.to_typescript(), verbose=verbose, callbacks=callbacks)
    if raw_response:
        response_chain = None
    else:
        response_chain = APIResponderChain.from_llm(llm, verbose=verbose, callbacks=callbacks)
    _requests = requests or Requests()
    return cls(api_request_chain=requests_chain, api_response_chain=response_chain, api_operation=operation, requests=_requests, param_mapping=param_mapping, verbose=verbose, return_intermediate_steps=return_intermediate_steps, callbacks=callbacks, **kwargs)