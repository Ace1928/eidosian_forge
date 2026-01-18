from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
def get_tracing_context(context: Optional[contextvars.Context]=None) -> Dict[str, Any]:
    """Get the current tracing context."""
    if context is None:
        return {'parent': _PARENT_RUN_TREE.get(), 'project_name': _PROJECT_NAME.get(), 'tags': _TAGS.get(), 'metadata': _METADATA.get()}
    return {k: context.get(v) for k, v in _CONTEXT_KEYS.items()}