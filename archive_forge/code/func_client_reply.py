import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_reply(self, reply: Union[Literal['ON'], Literal['OFF'], Literal['SKIP']], **kwargs) -> ResponseT:
    """
        Enable and disable redis server replies.

        ``reply`` Must be ON OFF or SKIP,
        ON - The default most with server replies to commands
        OFF - Disable server responses to commands
        SKIP - Skip the response of the immediately following command.

        Note: When setting OFF or SKIP replies, you will need a client object
        with a timeout specified in seconds, and will need to catch the
        TimeoutError.
        The test_client_reply unit test illustrates this, and
        conftest.py has a client with a timeout.

        See https://redis.io/commands/client-reply
        """
    replies = ['ON', 'OFF', 'SKIP']
    if reply not in replies:
        raise DataError(f'CLIENT REPLY must be one of {replies!r}')
    return self.execute_command('CLIENT REPLY', reply, **kwargs)