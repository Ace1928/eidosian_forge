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
def get_current_run_tree() -> Optional[run_trees.RunTree]:
    """Get the current run tree."""
    return _PARENT_RUN_TREE.get()