import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hget(self, name: str, key: str) -> Union[Awaitable[Optional[str]], Optional[str]]:
    """
        Return the value of ``key`` within the hash ``name``

        For more information see https://redis.io/commands/hget
        """
    return self.execute_command('HGET', name, key)