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
def sentinel_sentinels(self, service_name: str) -> Awaitable:
    """Returns a list of sentinels for ``service_name``"""
    return self.execute_command('SENTINEL SENTINELS', service_name)