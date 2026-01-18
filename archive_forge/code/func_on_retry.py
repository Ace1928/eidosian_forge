from __future__ import annotations
import asyncio
import functools
import logging
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import (
from uuid import UUID
from langsmith.run_helpers import get_run_tree_context
from tenacity import RetryCallState
from langchain_core.callbacks.base import (
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils.env import env_var_is_set
def on_retry(self, retry_state: RetryCallState, **kwargs: Any) -> None:
    handle_event(self.handlers, 'on_retry', 'ignore_retry', retry_state, run_id=self.run_id, parent_run_id=self.parent_run_id, tags=self.tags, **kwargs)