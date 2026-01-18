import asyncio
import datetime
import hashlib
import inspect
import re
import time as mod_time
import warnings
from typing import (
from aioredis.compat import Protocol, TypedDict
from aioredis.connection import (
from aioredis.exceptions import (
from aioredis.lock import Lock
from aioredis.utils import safe_str, str_if_bytes
def decr(self, name: KeyT, amount: int=1) -> Awaitable:
    """
        Decrements the value of ``key`` by ``amount``.  If no key exists,
        the value will be initialized as 0 - ``amount``
        """
    return self.decrby(name, amount)