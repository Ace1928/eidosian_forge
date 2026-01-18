import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_unblock(self, client_id: int, error: bool=False, **kwargs) -> ResponseT:
    """
        Unblocks a connection by its client id.
        If ``error`` is True, unblocks the client with a special error message.
        If ``error`` is False (default), the client is unblocked using the
        regular timeout mechanism.

        For more information see https://redis.io/commands/client-unblock
        """
    args = ['CLIENT UNBLOCK', int(client_id)]
    if error:
        args.append(b'ERROR')
    return self.execute_command(*args, **kwargs)