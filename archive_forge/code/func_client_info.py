import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_info(self, **kwargs) -> ResponseT:
    """
        Returns information and statistics about the current
        client connection.

        For more information see https://redis.io/commands/client-info
        """
    return self.execute_command('CLIENT INFO', **kwargs)