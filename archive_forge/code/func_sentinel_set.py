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
def sentinel_set(self, name: str, option: str, value: EncodableT) -> Awaitable:
    """Set Sentinel monitoring parameters for a given master"""
    return self.execute_command('SENTINEL SET', name, option, value)