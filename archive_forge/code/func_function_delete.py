import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def function_delete(self, library: str) -> Union[Awaitable[str], str]:
    """
        Delete the library called ``library`` and all its functions.

        For more information see https://redis.io/commands/function-delete
        """
    return self.execute_command('FUNCTION DELETE', library)