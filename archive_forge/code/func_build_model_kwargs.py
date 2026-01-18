from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.utils import build_extra_kwargs
@root_validator(pre=True)
def build_model_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """Build extra kwargs from additional params that were passed in."""
    all_required_field_names = get_pydantic_field_names(cls)
    extra = values.get('model_kwargs', {})
    values['model_kwargs'] = build_extra_kwargs(extra, values, all_required_field_names)
    return values