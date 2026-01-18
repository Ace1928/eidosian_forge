from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def end_all_processes(cls, verbose: bool=True, timeout: float=5.0):
    """
        Terminates all processes
        """
    for kind, names in cls.worker_processes.items():
        for name in names:
            cls.stop_worker_processes(name=name, verbose=verbose, timeout=timeout, kind=kind)
    cls.stop_server_processes(verbose=verbose, timeout=timeout)