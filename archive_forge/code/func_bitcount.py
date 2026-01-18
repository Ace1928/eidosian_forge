import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def bitcount(self, key: KeyT, start: Union[int, None]=None, end: Union[int, None]=None, mode: Optional[str]=None) -> ResponseT:
    """
        Returns the count of set bits in the value of ``key``.  Optional
        ``start`` and ``end`` parameters indicate which bytes to consider

        For more information see https://redis.io/commands/bitcount
        """
    params = [key]
    if start is not None and end is not None:
        params.append(start)
        params.append(end)
    elif start is not None and end is None or (end is not None and start is None):
        raise DataError('Both start and end must be specified')
    if mode is not None:
        params.append(mode)
    return self.execute_command('BITCOUNT', *params)