from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def register_on_close(cls, func: Callable, *args, **kwargs):
    """
        Registers a function to be called on close
        """
    import functools
    _func = functools.partial(func, *args, **kwargs)
    cls.on_close_funcs.append(_func)
    cls.logger.info(f'Registered function {func.__name__} to be called on close')