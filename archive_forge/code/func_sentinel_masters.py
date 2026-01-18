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
def sentinel_masters(self) -> Awaitable:
    """Returns a list of dictionaries containing each master's state."""
    return self.execute_command('SENTINEL MASTERS')