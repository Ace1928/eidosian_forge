import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def sinterstore(self, dest: str, keys: List, *args: List) -> Union[Awaitable[int], int]:
    """
        Store the intersection of sets specified by ``keys`` into a new
        set named ``dest``.  Returns the number of keys in the new set.

        For more information see https://redis.io/commands/sinterstore
        """
    args = list_or_args(keys, args)
    return self.execute_command('SINTERSTORE', dest, *args)