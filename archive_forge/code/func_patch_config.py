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
def patch_config(config: Optional[RunnableConfig], *, callbacks: Optional[BaseCallbackManager]=None, recursion_limit: Optional[int]=None, max_concurrency: Optional[int]=None, run_name: Optional[str]=None, configurable: Optional[Dict[str, Any]]=None) -> RunnableConfig:
    """Patch a config with new values.

    Args:
        config (Optional[RunnableConfig]): The config to patch.
        copy_locals (bool, optional): Whether to copy locals. Defaults to False.
        callbacks (Optional[BaseCallbackManager], optional): The callbacks to set.
          Defaults to None.
        recursion_limit (Optional[int], optional): The recursion limit to set.
          Defaults to None.
        max_concurrency (Optional[int], optional): The max concurrency to set.
          Defaults to None.
        run_name (Optional[str], optional): The run name to set. Defaults to None.
        configurable (Optional[Dict[str, Any]], optional): The configurable to set.
          Defaults to None.

    Returns:
        RunnableConfig: The patched config.
    """
    config = ensure_config(config)
    if callbacks is not None:
        config['callbacks'] = callbacks
        if 'run_name' in config:
            del config['run_name']
        if 'run_id' in config:
            del config['run_id']
    if recursion_limit is not None:
        config['recursion_limit'] = recursion_limit
    if max_concurrency is not None:
        config['max_concurrency'] = max_concurrency
    if run_name is not None:
        config['run_name'] = run_name
    if configurable is not None:
        config['configurable'] = {**config.get('configurable', {}), **configurable}
    return config