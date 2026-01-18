import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def memory_malloc_stats(self, **kwargs) -> ResponseT:
    """
        Return an internal statistics report from the memory allocator.

        See: https://redis.io/commands/memory-malloc-stats
        """
    return self.execute_command('MEMORY MALLOC-STATS', **kwargs)