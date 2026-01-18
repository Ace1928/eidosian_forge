from __future__ import annotations
import asyncio
import functools
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
def generate_with_retry(llm: Tongyi, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _generate_with_retry(**_kwargs: Any) -> Any:
        resp = llm.client.call(**_kwargs)
        return check_response(resp)
    return _generate_with_retry(**kwargs)