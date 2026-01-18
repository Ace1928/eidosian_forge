import json
import urllib.request
import warnings
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator, validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
@staticmethod
def escape_special_characters(prompt: str) -> str:
    """Escapes any special characters in `prompt`"""
    escape_map = {'\\': '\\\\', '"': '\\"', '\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
    for escape_sequence, escaped_sequence in escape_map.items():
        prompt = prompt.replace(escape_sequence, escaped_sequence)
    return prompt