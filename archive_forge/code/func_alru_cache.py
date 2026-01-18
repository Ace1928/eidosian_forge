import random
import inspect
import backoff
import functools
import aiohttpx
from typing import Callable, List, Optional, TypeVar, TYPE_CHECKING
def alru_cache(*args, **kwargs):

    def decorator(func: Callable[PT, RT]) -> Callable[PT, RT]:
        return func
    return decorator