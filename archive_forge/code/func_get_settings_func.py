from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def get_settings_func(cls, func: Union[Callable, str]) -> Callable:
    """
        Returns the settings func
        """
    if isinstance(func, str):
        func = getattr(cls.settings, func)
    return func