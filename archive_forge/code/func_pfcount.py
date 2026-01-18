import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def pfcount(self, *sources: KeyT) -> ResponseT:
    """
        Return the approximated cardinality of
        the set observed by the HyperLogLog at key(s).

        For more information see https://redis.io/commands/pfcount
        """
    return self.execute_command('PFCOUNT', *sources)