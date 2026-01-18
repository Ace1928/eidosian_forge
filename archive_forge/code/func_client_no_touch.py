import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_no_touch(self, mode: str) -> Union[Awaitable[str], str]:
    """
        # The command controls whether commands sent by the client will alter
        # the LRU/LFU of the keys they access.
        # When turned on, the current client will not change LFU/LRU stats,
        # unless it sends the TOUCH command.

        For more information see https://redis.io/commands/client-no-touch
        """
    return self.execute_command('CLIENT NO-TOUCH', mode)