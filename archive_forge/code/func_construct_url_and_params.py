import os
import sys
from typing import Any, Dict, Optional, Tuple
import aiohttp
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
    params = self.get_params(query)
    params['source'] = 'python'
    if self.serpapi_api_key:
        params['serp_api_key'] = self.serpapi_api_key
    params['output'] = 'json'
    url = 'https://serpapi.com/search'
    return (url, params)