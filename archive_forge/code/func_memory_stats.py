import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def memory_stats(self, **kwargs) -> ResponseT:
    """
        Return a dictionary of memory stats

        For more information see https://redis.io/commands/memory-stats
        """
    return self.execute_command('MEMORY STATS', **kwargs)