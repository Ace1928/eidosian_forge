import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hset(self, name: str, key: Optional[str]=None, value: Optional[str]=None, mapping: Optional[dict]=None, items: Optional[list]=None) -> Union[Awaitable[int], int]:
    """
        Set ``key`` to ``value`` within hash ``name``,
        ``mapping`` accepts a dict of key/value pairs that will be
        added to hash ``name``.
        ``items`` accepts a list of key/value pairs that will be
        added to hash ``name``.
        Returns the number of fields that were added.

        For more information see https://redis.io/commands/hset
        """
    if key is None and (not mapping) and (not items):
        raise DataError("'hset' with no key value pairs")
    pieces = []
    if items:
        pieces.extend(items)
    if key is not None:
        pieces.extend((key, value))
    if mapping:
        for pair in mapping.items():
            pieces.extend(pair)
    return self.execute_command('HSET', name, *pieces)