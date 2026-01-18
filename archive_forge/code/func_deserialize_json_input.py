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
def deserialize_json_input(self, serialized_args: str) -> dict:
    """Use the serialized typescript dictionary.

        Resolve the path, query params dict, and optional requestBody dict.
        """
    args: dict = json.loads(serialized_args)
    path = self._construct_path(args)
    body_params = self._extract_body_params(args)
    query_params = self._extract_query_params(args)
    return {'url': path, 'data': body_params, 'params': query_params}