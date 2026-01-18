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
def should_cache_value(self, val: Any) -> bool:
    """
        Returns whether or not the value should be cached
        """
    if self.exclude_null and val is None:
        return False
    if self.exclude_exceptions:
        if isinstance(self.exclude_exceptions, list):
            return not isinstance(val, tuple(self.exclude_exceptions))
        if isinstance(val, Exception):
            return False
    return True