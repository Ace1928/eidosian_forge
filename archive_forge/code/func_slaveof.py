import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def slaveof(self, host: Union[str, None]=None, port: Union[int, None]=None, **kwargs) -> ResponseT:
    """
        Set the server to be a replicated slave of the instance identified
        by the ``host`` and ``port``. If called without arguments, the
        instance is promoted to a master instead.

        For more information see https://redis.io/commands/slaveof
        """
    if host is None and port is None:
        return self.execute_command('SLAVEOF', b'NO', b'ONE', **kwargs)
    return self.execute_command('SLAVEOF', host, port, **kwargs)