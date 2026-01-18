import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def swapdb(self, first: int, second: int, **kwargs) -> ResponseT:
    """
        Swap two databases

        For more information see https://redis.io/commands/swapdb
        """
    return self.execute_command('SWAPDB', first, second, **kwargs)