import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def msetnx(self, mapping: Mapping[AnyKeyT, EncodableT]) -> ResponseT:
    """
        Sets key/values based on a mapping if none of the keys are already set.
        Mapping is a dictionary of key/value pairs. Both keys and values
        should be strings or types that can be cast to a string via str().
        Returns a boolean indicating if the operation was successful.

        For more information see https://redis.io/commands/msetnx
        """
    items = []
    for pair in mapping.items():
        items.extend(pair)
    return self.execute_command('MSETNX', *items)