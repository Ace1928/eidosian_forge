import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def rpush(self, name: str, *values: FieldT) -> Union[Awaitable[int], int]:
    """
        Push ``values`` onto the tail of the list ``name``

        For more information see https://redis.io/commands/rpush
        """
    return self.execute_command('RPUSH', name, *values)