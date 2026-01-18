import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def bgsave(self, schedule: bool=True, **kwargs) -> ResponseT:
    """
        Tell the Redis server to save its data to disk.  Unlike save(),
        this method is asynchronous and returns immediately.

        For more information see https://redis.io/commands/bgsave
        """
    pieces = []
    if schedule:
        pieces.append('SCHEDULE')
    return self.execute_command('BGSAVE', *pieces, **kwargs)