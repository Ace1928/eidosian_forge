import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_pause(self, timeout: int, all: bool=True, **kwargs) -> ResponseT:
    """
        Suspend all the Redis clients for the specified amount of time.


        For more information see https://redis.io/commands/client-pause

        :param timeout: milliseconds to pause clients
        :param all: If true (default) all client commands are blocked.
        otherwise, clients are only blocked if they attempt to execute
        a write command.
        For the WRITE mode, some commands have special behavior:
        EVAL/EVALSHA: Will block client for all scripts.
        PUBLISH: Will block client.
        PFCOUNT: Will block client.
        WAIT: Acknowledgments will be delayed, so this command will
        appear blocked.
        """
    args = ['CLIENT PAUSE', str(timeout)]
    if not isinstance(timeout, int):
        raise DataError('CLIENT PAUSE timeout must be an integer')
    if not all:
        args.append('WRITE')
    return self.execute_command(*args, **kwargs)