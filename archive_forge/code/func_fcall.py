import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def fcall(self, function, numkeys: int, *keys_and_args: Optional[List]) -> Union[Awaitable[str], str]:
    """
        Invoke a function.

        For more information see https://redis.io/commands/fcall
        """
    return self._fcall('FCALL', function, numkeys, *keys_and_args)