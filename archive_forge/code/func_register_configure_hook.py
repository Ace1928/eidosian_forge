from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
from uuid import UUID
from langsmith import utils as ls_utils
from langsmith.run_helpers import get_run_tree_context
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.tracers.schemas import TracerSessionV1
from langchain_core.utils.env import env_var_is_set
def register_configure_hook(context_var: ContextVar[Optional[Any]], inheritable: bool, handle_class: Optional[Type[BaseCallbackHandler]]=None, env_var: Optional[str]=None) -> None:
    """Register a configure hook.

    Args:
        context_var (ContextVar[Optional[Any]]): The context variable.
        inheritable (bool): Whether the context variable is inheritable.
        handle_class (Optional[Type[BaseCallbackHandler]], optional):
          The callback handler class. Defaults to None.
        env_var (Optional[str], optional): The environment variable. Defaults to None.

    Raises:
        ValueError: If env_var is set, handle_class must also be set
          to a non-None value.
    """
    if env_var is not None and handle_class is None:
        raise ValueError('If env_var is set, handle_class must also be set to a non-None value.')
    from langchain_core.callbacks.base import BaseCallbackHandler
    _configure_hooks.append((cast(ContextVar[Optional[BaseCallbackHandler]], context_var), inheritable, handle_class, env_var))