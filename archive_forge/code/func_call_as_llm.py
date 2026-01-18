from __future__ import annotations
import asyncio
import inspect
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import (
from langchain_core._api import deprecated
from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import BaseLanguageModel, LanguageModelInput
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.tracers.log_stream import LogStreamCallbackHandler
@deprecated('0.1.7', alternative='invoke', removal='0.2.0')
def call_as_llm(self, message: str, stop: Optional[List[str]]=None, **kwargs: Any) -> str:
    return self.predict(message, stop=stop, **kwargs)