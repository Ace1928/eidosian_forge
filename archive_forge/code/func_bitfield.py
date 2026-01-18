import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def bitfield(self: Union['Redis', 'AsyncRedis'], key: KeyT, default_overflow: Union[str, None]=None) -> BitFieldOperation:
    """
        Return a BitFieldOperation instance to conveniently construct one or
        more bitfield operations on ``key``.

        For more information see https://redis.io/commands/bitfield
        """
    return BitFieldOperation(self, key, default_overflow=default_overflow)