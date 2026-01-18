import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def eval_ro(self, script: str, numkeys: int, *keys_and_args: str) -> Union[Awaitable[str], str]:
    """
        The read-only variant of the EVAL command

        Execute the read-only Lua ``script`` specifying the ``numkeys`` the script
        will touch and the key names and argument values in ``keys_and_args``.
        Returns the result of the script.

        For more information see  https://redis.io/commands/eval_ro
        """
    return self._eval('EVAL_RO', script, numkeys, *keys_and_args)