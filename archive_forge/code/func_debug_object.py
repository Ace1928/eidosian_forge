import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def debug_object(self, key: KeyT, **kwargs) -> ResponseT:
    """
        Returns version specific meta information about a given key

        For more information see https://redis.io/commands/debug-object
        """
    return self.execute_command('DEBUG OBJECT', key, **kwargs)