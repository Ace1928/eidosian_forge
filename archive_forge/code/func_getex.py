import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def getex(self, name: KeyT, ex: Union[ExpiryT, None]=None, px: Union[ExpiryT, None]=None, exat: Union[AbsExpiryT, None]=None, pxat: Union[AbsExpiryT, None]=None, persist: bool=False) -> ResponseT:
    """
        Get the value of key and optionally set its expiration.
        GETEX is similar to GET, but is a write command with
        additional options. All time parameters can be given as
        datetime.timedelta or integers.

        ``ex`` sets an expire flag on key ``name`` for ``ex`` seconds.

        ``px`` sets an expire flag on key ``name`` for ``px`` milliseconds.

        ``exat`` sets an expire flag on key ``name`` for ``ex`` seconds,
        specified in unix time.

        ``pxat`` sets an expire flag on key ``name`` for ``ex`` milliseconds,
        specified in unix time.

        ``persist`` remove the time to live associated with ``name``.

        For more information see https://redis.io/commands/getex
        """
    opset = {ex, px, exat, pxat}
    if len(opset) > 2 or (len(opset) > 1 and persist):
        raise DataError('``ex``, ``px``, ``exat``, ``pxat``, and ``persist`` are mutually exclusive.')
    pieces: list[EncodableT] = []
    if ex is not None:
        pieces.append('EX')
        if isinstance(ex, datetime.timedelta):
            ex = int(ex.total_seconds())
        pieces.append(ex)
    if px is not None:
        pieces.append('PX')
        if isinstance(px, datetime.timedelta):
            px = int(px.total_seconds() * 1000)
        pieces.append(px)
    if exat is not None:
        pieces.append('EXAT')
        if isinstance(exat, datetime.datetime):
            exat = int(exat.timestamp())
        pieces.append(exat)
    if pxat is not None:
        pieces.append('PXAT')
        if isinstance(pxat, datetime.datetime):
            pxat = int(pxat.timestamp() * 1000)
        pieces.append(pxat)
    if persist:
        pieces.append('PERSIST')
    return self.execute_command('GETEX', name, *pieces)