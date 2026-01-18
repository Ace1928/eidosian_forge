import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def lmpop(self, num_keys: int, *args: List[str], direction: str, count: Optional[int]=1) -> Union[Awaitable[list], list]:
    """
        Pop ``count`` values (default 1) first non-empty list key from the list
        of args provided key names.

        For more information see https://redis.io/commands/lmpop
        """
    args = [num_keys] + list(args) + [direction]
    if count != 1:
        args.extend(['COUNT', count])
    return self.execute_command('LMPOP', *args)