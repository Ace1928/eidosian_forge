import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def function_kill(self) -> Union[Awaitable[str], str]:
    """
        Kill a function that is currently executing.

        For more information see https://redis.io/commands/function-kill
        """
    return self.execute_command('FUNCTION KILL')