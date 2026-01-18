import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hincrby(self, name: str, key: str, amount: int=1) -> Union[Awaitable[int], int]:
    """
        Increment the value of ``key`` in hash ``name`` by ``amount``

        For more information see https://redis.io/commands/hincrby
        """
    return self.execute_command('HINCRBY', name, key, amount)