import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def spop(self, name: str, count: Optional[int]=None) -> Union[str, List, None]:
    """
        Remove and return a random member of set ``name``

        For more information see https://redis.io/commands/spop
        """
    args = count is not None and [count] or []
    return self.execute_command('SPOP', name, *args)