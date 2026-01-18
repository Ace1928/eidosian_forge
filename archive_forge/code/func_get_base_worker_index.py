from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def get_base_worker_index() -> int:
    """
    Gets the base worker index
    """
    global _base_worker_index
    if _base_worker_index is None:
        from lazyops.utils.system import is_in_kubernetes, get_host_name
        if is_in_kubernetes() and get_host_name()[-1].isdigit():
            _base_worker_index = int(get_host_name()[-1])
        else:
            _base_worker_index = 0
    return _base_worker_index