from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def set_queue(cls, name: str, queue: 'TaskQueue', kind: Optional[str]=None):
    """
        Set a queue
        """
    if kind is None:
        kind = 'default'
    if kind not in cls.queues:
        cls.add_queue_type(kind)
    cls.queues[kind][name] = queue