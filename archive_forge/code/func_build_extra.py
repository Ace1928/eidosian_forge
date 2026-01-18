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
@root_validator(pre=True)
def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """Build extra kwargs from additional params that were passed in."""
    all_required_field_names = {field.alias for field in cls.__fields__.values()}
    extra = values.get('model_kwargs', {})
    for field_name in list(values):
        if field_name not in all_required_field_names:
            if field_name in extra:
                raise ValueError(f'Found {field_name} supplied twice.')
            extra[field_name] = values.pop(field_name)
    values['model_kwargs'] = extra
    return values