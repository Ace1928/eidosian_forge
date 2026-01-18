import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def command_getkeysandflags(self, *args: List[str]) -> List[Union[str, List[str]]]:
    """
        Returns array of keys from a full Redis command and their usage flags.

        For more information see https://redis.io/commands/command-getkeysandflags
        """
    return self.execute_command('COMMAND GETKEYSANDFLAGS', *args)