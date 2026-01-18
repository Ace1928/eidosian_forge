import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def bitpos(self, key: KeyT, bit: int, start: Union[int, None]=None, end: Union[int, None]=None, mode: Optional[str]=None) -> ResponseT:
    """
        Return the position of the first bit set to 1 or 0 in a string.
        ``start`` and ``end`` defines search range. The range is interpreted
        as a range of bytes and not a range of bits, so start=0 and end=2
        means to look at the first three bytes.

        For more information see https://redis.io/commands/bitpos
        """
    if bit not in (0, 1):
        raise DataError('bit must be 0 or 1')
    params = [key, bit]
    start is not None and params.append(start)
    if start is not None and end is not None:
        params.append(end)
    elif start is None and end is not None:
        raise DataError('start argument is not set, when end is specified')
    if mode is not None:
        params.append(mode)
    return self.execute_command('BITPOS', *params)