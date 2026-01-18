import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def lpop(self, name: str, count: Optional[int]=None) -> Union[Awaitable[Union[str, List, None]], Union[str, List, None]]:
    """
        Removes and returns the first elements of the list ``name``.

        By default, the command pops a single element from the beginning of
        the list. When provided with the optional ``count`` argument, the reply
        will consist of up to count elements, depending on the list's length.

        For more information see https://redis.io/commands/lpop
        """
    if count is not None:
        return self.execute_command('LPOP', name, count)
    else:
        return self.execute_command('LPOP', name)