import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def scan_iter(self, match: Union[PatternT, None]=None, count: Union[int, None]=None, _type: Union[str, None]=None, **kwargs) -> Iterator:
    """
        Make an iterator using the SCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` provides a hint to Redis about the number of keys to
            return per batch.

        ``_type`` filters the returned values by a particular Redis type.
            Stock Redis instances allow for the following types:
            HASH, LIST, SET, STREAM, STRING, ZSET
            Additionally, Redis modules can expose other types as well.
        """
    cursor = '0'
    while cursor != 0:
        cursor, data = self.scan(cursor=cursor, match=match, count=count, _type=_type, **kwargs)
        yield from data