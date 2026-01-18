import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def hexists(self, name: str, key: str) -> Union[Awaitable[bool], bool]:
    """
        Returns a boolean indicating if ``key`` exists within hash ``name``

        For more information see https://redis.io/commands/hexists
        """
    return self.execute_command('HEXISTS', name, key)