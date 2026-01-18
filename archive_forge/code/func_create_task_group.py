from __future__ import annotations
import math
from collections.abc import Generator
from contextlib import contextmanager
from types import TracebackType
from ..abc._tasks import TaskGroup, TaskStatus
from ._eventloop import get_async_backend
def create_task_group() -> TaskGroup:
    """
    Create a task group.

    :return: a task group

    """
    return get_async_backend().create_task_group()