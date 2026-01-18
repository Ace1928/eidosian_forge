import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def script_load(self, script: ScriptTextT) -> ResponseT:
    """
        Load a Lua ``script`` into the script cache. Returns the SHA.

        For more information see https://redis.io/commands/script-load
        """
    return self.execute_command('SCRIPT LOAD', script)