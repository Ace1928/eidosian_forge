import asyncio
import functools
import hashlib
import os
from typing import Callable, Optional
import cloudpickle
from diskcache import Cache
def hash_arguments(*args, **kwargs) -> str:
    """Create a hash out of the args and kwargs provided"""
    result = hashlib.md5()
    for item in list(args) + sorted(kwargs.items()):
        result.update(cloudpickle.dumps(item))
    return result.hexdigest()