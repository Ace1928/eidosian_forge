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
def create_llm_result(self, choices: Any, prompts: List[str], params: Dict[str, Any], token_usage: Dict[str, int], *, system_fingerprint: Optional[str]=None) -> LLMResult:
    """Create the LLMResult from the choices and prompts."""
    generations = []
    n = params.get('n', self.n)
    for i, _ in enumerate(prompts):
        sub_choices = choices[i * n:(i + 1) * n]
        generations.append([Generation(text=choice['text'], generation_info=dict(finish_reason=choice.get('finish_reason'), logprobs=choice.get('logprobs'))) for choice in sub_choices])
    llm_output = {'token_usage': token_usage, 'model_name': self.model_name}
    if system_fingerprint:
        llm_output['system_fingerprint'] = system_fingerprint
    return LLMResult(generations=generations, llm_output=llm_output)