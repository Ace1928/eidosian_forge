import logging
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
def get_count_value(key: str, result: Dict[str, Any]) -> int:
    return result.get(key, 0) or 0