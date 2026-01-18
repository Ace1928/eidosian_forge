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
@contextmanager
def get_executor_for_config(config: Optional[RunnableConfig]) -> Generator[Executor, None, None]:
    """Get an executor for a config.

    Args:
        config (RunnableConfig): The config.

    Yields:
        Generator[Executor, None, None]: The executor.
    """
    config = config or {}
    with ContextThreadPoolExecutor(max_workers=config.get('max_concurrency')) as executor:
        yield executor