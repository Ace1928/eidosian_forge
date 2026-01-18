from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms import BaseLLM
from langchain_community.utilities.vertexai import create_retry_decorator
@property
def is_gemini(self) -> bool:
    """Returns whether a model is belongs to a Gemini family or not."""
    return _is_gemini_model(self.model_name)