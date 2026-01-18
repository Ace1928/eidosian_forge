import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def evalsha_ro(self, sha: str, numkeys: int, *keys_and_args: str) -> Union[Awaitable[str], str]:
    """
        The read-only variant of the EVALSHA command

        Use the ``sha`` to execute a read-only Lua script already registered via EVAL
        or SCRIPT LOAD. Specify the ``numkeys`` the script will touch and the
        key names and argument values in ``keys_and_args``. Returns the result
        of the script.

        For more information see  https://redis.io/commands/evalsha_ro
        """
    return self._evalsha('EVALSHA_RO', sha, numkeys, *keys_and_args)