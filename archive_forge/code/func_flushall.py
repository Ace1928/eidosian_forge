import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def flushall(self, asynchronous: bool=False, **kwargs) -> ResponseT:
    """
        Delete all keys in all databases on the current host.

        ``asynchronous`` indicates whether the operation is
        executed asynchronously by the server.

        For more information see https://redis.io/commands/flushall
        """
    args = []
    if asynchronous:
        args.append(b'ASYNC')
    return self.execute_command('FLUSHALL', *args, **kwargs)