import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def memory_purge(self, **kwargs) -> ResponseT:
    """
        Attempts to purge dirty pages for reclamation by allocator

        For more information see https://redis.io/commands/memory-purge
        """
    return self.execute_command('MEMORY PURGE', **kwargs)