import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def function_stats(self) -> Union[Awaitable[List], List]:
    """
        Return information about the function that's currently running
        and information about the available execution engines.

        For more information see https://redis.io/commands/function-stats
        """
    return self.execute_command('FUNCTION STATS')