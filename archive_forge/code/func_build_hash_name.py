from __future__ import annotations
import time
import anyio
import inspect
import contextlib 
import functools
import hashlib
from lazyops.types.common import UpperStrEnum
from lazyops.utils import timed_cache
from lazyops.utils.helpers import create_background_task, fail_after
from lazyops.utils.lazy import lazy_import
from lazyops.utils.pooler import ThreadPooler
from lazyops.utils.lazy import get_function_name
from .compat import BaseModel, root_validator, get_pyd_dict
from .base import ENOVAL
from typing import Optional, Dict, Any, Callable, List, Union, TypeVar, Type, overload, TYPE_CHECKING
from aiokeydb.utils.logs import logger
from aiokeydb.utils.helpers import afail_after
def build_hash_name(self, func: Callable, *args, **kwargs) -> str:
    """
        Builds the name for the function
        """
    if self.cache_field is not None:
        return self.cache_field
    if self.name:
        self.cache_field = self.name(func, *args, **kwargs) if callable(self.name) else self.name
    else:
        func = inspect.unwrap(func)
        self.cache_field = f'{func.__module__}.{func.__qualname__}'
    return self.cache_field