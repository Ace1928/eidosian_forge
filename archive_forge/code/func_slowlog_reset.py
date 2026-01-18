import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def slowlog_reset(self, **kwargs) -> ResponseT:
    """
        Remove all items in the slowlog

        For more information see https://redis.io/commands/slowlog-reset
        """
    return self.execute_command('SLOWLOG RESET', **kwargs)