import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def brpoplpush(self, src: str, dst: str, timeout: Optional[int]=0) -> Union[Awaitable[Optional[str]], Optional[str]]:
    """
        Pop a value off the tail of ``src``, push it on the head of ``dst``
        and then return it.

        This command blocks until a value is in ``src`` or until ``timeout``
        seconds elapse, whichever is first. A ``timeout`` value of 0 blocks
        forever.

        For more information see https://redis.io/commands/brpoplpush
        """
    if timeout is None:
        timeout = 0
    return self.execute_command('BRPOPLPUSH', src, dst, timeout)