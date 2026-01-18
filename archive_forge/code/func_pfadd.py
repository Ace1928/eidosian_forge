import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def pfadd(self, name: KeyT, *values: FieldT) -> ResponseT:
    """
        Adds the specified elements to the specified HyperLogLog.

        For more information see https://redis.io/commands/pfadd
        """
    return self.execute_command('PFADD', name, *values)