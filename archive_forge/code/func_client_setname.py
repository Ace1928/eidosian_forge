import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_setname(self, name: str, **kwargs) -> ResponseT:
    """
        Sets the current connection name

        For more information see https://redis.io/commands/client-setname

        .. note::
           This method sets client name only for **current** connection.

           If you want to set a common name for all connections managed
           by this client, use ``client_name`` constructor argument.
        """
    return self.execute_command('CLIENT SETNAME', name, **kwargs)