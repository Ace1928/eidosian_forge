import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def geopos(self, name: KeyT, *values: FieldT) -> ResponseT:
    """
        Return the positions of each item of ``values`` as members of
        the specified key identified by the ``name`` argument. Each position
        is represented by the pairs lon and lat.

        For more information see https://redis.io/commands/geopos
        """
    return self.execute_command('GEOPOS', name, *values)