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
def get_callback_manager_for_config(config: RunnableConfig) -> CallbackManager:
    """Get a callback manager for a config.

    Args:
        config (RunnableConfig): The config.

    Returns:
        CallbackManager: The callback manager.
    """
    from langchain_core.callbacks.manager import CallbackManager
    return CallbackManager.configure(inheritable_callbacks=config.get('callbacks'), inheritable_tags=config.get('tags'), inheritable_metadata=config.get('metadata'))