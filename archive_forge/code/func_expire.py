import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def expire(self, name: KeyT, time: ExpiryT, nx: bool=False, xx: bool=False, gt: bool=False, lt: bool=False) -> ResponseT:
    """
        Set an expire flag on key ``name`` for ``time`` seconds with given
        ``option``. ``time`` can be represented by an integer or a Python timedelta
        object.

        Valid options are:
            NX -> Set expiry only when the key has no expiry
            XX -> Set expiry only when the key has an existing expiry
            GT -> Set expiry only when the new expiry is greater than current one
            LT -> Set expiry only when the new expiry is less than current one

        For more information see https://redis.io/commands/expire
        """
    if isinstance(time, datetime.timedelta):
        time = int(time.total_seconds())
    exp_option = list()
    if nx:
        exp_option.append('NX')
    if xx:
        exp_option.append('XX')
    if gt:
        exp_option.append('GT')
    if lt:
        exp_option.append('LT')
    return self.execute_command('EXPIRE', name, time, *exp_option)