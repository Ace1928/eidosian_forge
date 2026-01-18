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
def run_post_call_hook(self, result: Any, *args, is_hit: Optional[bool]=None, **kwargs) -> None:
    """
        Runs the post call hook which fires after the function is called
        """
    if not self.has_post_call_hook:
        return
    if self.super_verbose:
        logger.info(f'[{self.cache_field}] Running Post Call Hook')
    create_background_task(self.post_call_hook, result, *args, is_hit=is_hit, **kwargs)