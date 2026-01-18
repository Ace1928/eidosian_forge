import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def sunion(self, keys: List, *args: List) -> Union[Awaitable[List], List]:
    """
        Return the union of sets specified by ``keys``

        For more information see https://redis.io/commands/sunion
        """
    args = list_or_args(keys, args)
    return self.execute_command('SUNION', *args)