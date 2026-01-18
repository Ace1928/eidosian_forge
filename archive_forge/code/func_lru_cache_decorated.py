import asyncio
import functools
from typing import Tuple
@functools.lru_cache()
def lru_cache_decorated(arg1):
    return arg1