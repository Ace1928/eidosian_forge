import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def readwrite(self, **kwargs) -> ResponseT:
    """
        Disables read queries for a connection to a Redis Cluster slave node.

        For more information see https://redis.io/commands/readwrite
        """
    return self.execute_command('READWRITE', **kwargs)