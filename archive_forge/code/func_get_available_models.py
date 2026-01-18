from __future__ import annotations
import logging
import os
import sys
from typing import TYPE_CHECKING, Dict, Optional, Set
import requests
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.adapters.openai import convert_message_to_dict
from langchain_community.chat_models.openai import (
from langchain_community.utils.openai import is_openai_v1
@staticmethod
def get_available_models(anyscale_api_key: Optional[str]=None, anyscale_api_base: str=DEFAULT_API_BASE) -> Set[str]:
    """Get available models from Anyscale API."""
    try:
        anyscale_api_key = anyscale_api_key or os.environ['ANYSCALE_API_KEY']
    except KeyError as e:
        raise ValueError('Anyscale API key must be passed as keyword argument or set in environment variable ANYSCALE_API_KEY.') from e
    models_url = f'{anyscale_api_base}/models'
    models_response = requests.get(models_url, headers={'Authorization': f'Bearer {anyscale_api_key}'})
    if models_response.status_code != 200:
        raise ValueError(f'Error getting models from {models_url}: {models_response.status_code}')
    return {model['id'] for model in models_response.json()['data']}