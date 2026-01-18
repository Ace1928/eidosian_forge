from __future__ import annotations
import logging
import os
import sys
import warnings
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from langchain_core.utils.utils import build_extra_kwargs
from langchain_community.utils.openai import is_openai_v1
def completion_with_retry(llm: Union[BaseOpenAI, OpenAIChat], run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    if is_openai_v1():
        return llm.client.create(**kwargs)
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return llm.client.create(**kwargs)
    return _completion_with_retry(**kwargs)