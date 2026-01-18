import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_no_evict(self, mode: str) -> Union[Awaitable[str], str]:
    """
        Sets the client eviction mode for the current connection.

        For more information see https://redis.io/commands/client-no-evict
        """
    return self.execute_command('CLIENT NO-EVICT', mode)