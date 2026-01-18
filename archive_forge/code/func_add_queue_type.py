from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def add_queue_type(cls, kind: str):
    """
        Add a queue type
        """
    if kind not in cls.queues:
        setattr(cls, kind, {})
        cls.queues[kind] = getattr(cls, kind)