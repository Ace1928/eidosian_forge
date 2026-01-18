import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def bgrewriteaof(self, **kwargs):
    """Tell the Redis server to rewrite the AOF file from data in memory.

        For more information see https://redis.io/commands/bgrewriteaof
        """
    return self.execute_command('BGREWRITEAOF', **kwargs)