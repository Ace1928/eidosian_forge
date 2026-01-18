import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def quit(self, **kwargs) -> ResponseT:
    """
        Ask the server to close the connection.

        For more information see https://redis.io/commands/quit
        """
    return self.execute_command('QUIT', **kwargs)