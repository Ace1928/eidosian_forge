import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def rpushx(self, name: str, *values: str) -> Union[Awaitable[int], int]:
    """
        Push ``value`` onto the tail of the list ``name`` if ``name`` exists

        For more information see https://redis.io/commands/rpushx
        """
    return self.execute_command('RPUSHX', name, *values)