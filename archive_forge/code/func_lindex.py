import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def lindex(self, name: str, index: int) -> Union[Awaitable[Optional[str]], Optional[str]]:
    """
        Return the item from list ``name`` at position ``index``

        Negative indexes are supported and will return an item at the
        end of the list

        For more information see https://redis.io/commands/lindex
        """
    return self.execute_command('LINDEX', name, index)