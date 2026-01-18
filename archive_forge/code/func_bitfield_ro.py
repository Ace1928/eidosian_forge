import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def bitfield_ro(self: Union['Redis', 'AsyncRedis'], key: KeyT, encoding: str, offset: BitfieldOffsetT, items: Optional[list]=None) -> ResponseT:
    """
        Return an array of the specified bitfield values
        where the first value is found using ``encoding`` and ``offset``
        parameters and remaining values are result of corresponding
        encoding/offset pairs in optional list ``items``
        Read-only variant of the BITFIELD command.

        For more information see https://redis.io/commands/bitfield_ro
        """
    params = [key, 'GET', encoding, offset]
    items = items or []
    for encoding, offset in items:
        params.extend(['GET', encoding, offset])
    return self.execute_command('BITFIELD_RO', *params)