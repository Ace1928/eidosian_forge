import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def fcall_ro(self, function, numkeys: int, *keys_and_args: Optional[List]) -> Union[Awaitable[str], str]:
    """
        This is a read-only variant of the FCALL command that cannot
        execute commands that modify data.

        For more information see https://redis.io/commands/fcal_ro
        """
    return self._fcall('FCALL_RO', function, numkeys, *keys_and_args)