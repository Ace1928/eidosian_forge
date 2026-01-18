import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def lpush(self, name: str, *values: FieldT) -> Union[Awaitable[int], int]:
    """
        Push ``values`` onto the head of the list ``name``

        For more information see https://redis.io/commands/lpush
        """
    return self.execute_command('LPUSH', name, *values)