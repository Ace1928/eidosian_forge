import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def sscan(self, name: KeyT, cursor: int=0, match: Union[PatternT, None]=None, count: Union[int, None]=None) -> ResponseT:
    """
        Incrementally return lists of elements in a set. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        For more information see https://redis.io/commands/sscan
        """
    pieces: list[EncodableT] = [name, cursor]
    if match is not None:
        pieces.extend([b'MATCH', match])
    if count is not None:
        pieces.extend([b'COUNT', count])
    return self.execute_command('SSCAN', *pieces)