import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_setinfo(self, attr: str, value: str, **kwargs) -> ResponseT:
    """
        Sets the current connection library name or version
        For mor information see https://redis.io/commands/client-setinfo
        """
    return self.execute_command('CLIENT SETINFO', attr, value, **kwargs)