import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def function_load(self, code: str, replace: Optional[bool]=False) -> Union[Awaitable[str], str]:
    """
        Load a library to Redis.
        :param code: the source code (must start with
        Shebang statement that provides a metadata about the library)
        :param replace: changes the behavior to overwrite the existing library
        with the new contents.
        Return the library name that was loaded.

        For more information see https://redis.io/commands/function-load
        """
    pieces = ['REPLACE'] if replace else []
    pieces.append(code)
    return self.execute_command('FUNCTION LOAD', *pieces)