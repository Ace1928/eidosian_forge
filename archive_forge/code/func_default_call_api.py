from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities.openapi import OpenAPISpec
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.utils.input import get_colored_text
from requests import Response
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.tools import APIOperation
def default_call_api(name: str, fn_args: dict, headers: Optional[dict]=None, params: Optional[dict]=None, **kwargs: Any) -> Any:
    method = _name_to_call_map[name]['method']
    url = _name_to_call_map[name]['url']
    path_params = fn_args.pop('path_params', {})
    url = _format_url(url, path_params)
    if 'data' in fn_args and isinstance(fn_args['data'], dict):
        fn_args['data'] = json.dumps(fn_args['data'])
    _kwargs = {**fn_args, **kwargs}
    if headers is not None:
        if 'headers' in _kwargs:
            _kwargs['headers'].update(headers)
        else:
            _kwargs['headers'] = headers
    if params is not None:
        if 'params' in _kwargs:
            _kwargs['params'].update(params)
        else:
            _kwargs['params'] = params
    return requests.request(method, url, **_kwargs)