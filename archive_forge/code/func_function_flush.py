import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def function_flush(self, mode: str='SYNC') -> Union[Awaitable[str], str]:
    """
        Deletes all the libraries.

        For more information see https://redis.io/commands/function-flush
        """
    return self.execute_command('FUNCTION FLUSH', mode)