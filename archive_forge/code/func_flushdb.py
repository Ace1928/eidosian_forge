import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def flushdb(self, asynchronous: bool=False, **kwargs) -> ResponseT:
    """
        Delete all keys in the current database.

        ``asynchronous`` indicates whether the operation is
        executed asynchronously by the server.

        For more information see https://redis.io/commands/flushdb
        """
    args = []
    if asynchronous:
        args.append(b'ASYNC')
    return self.execute_command('FLUSHDB', *args, **kwargs)