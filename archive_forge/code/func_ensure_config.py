from __future__ import annotations
import asyncio
import uuid
import warnings
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import partial
from typing import (
from typing_extensions import ParamSpec, TypedDict
from langchain_core.runnables.utils import (
def ensure_config(config: Optional[RunnableConfig]=None) -> RunnableConfig:
    """Ensure that a config is a dict with all keys present.

    Args:
        config (Optional[RunnableConfig], optional): The config to ensure.
          Defaults to None.

    Returns:
        RunnableConfig: The ensured config.
    """
    empty = RunnableConfig(tags=[], metadata={}, callbacks=None, recursion_limit=25, run_id=None)
    if (var_config := var_child_runnable_config.get()):
        empty.update(cast(RunnableConfig, {k: v for k, v in var_config.items() if v is not None}))
    if config is not None:
        empty.update(cast(RunnableConfig, {k: v for k, v in config.items() if v is not None}))
    for key, value in empty.get('configurable', {}).items():
        if isinstance(value, (str, int, float, bool)) and key not in empty['metadata']:
            empty['metadata'][key] = value
    return empty