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
def sentinel_master(self, service_name: str) -> Awaitable:
    """Returns a dictionary containing the specified masters state."""
    return self.execute_command('SENTINEL MASTER', service_name)