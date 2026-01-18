import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def module_load(self, path, *args) -> ResponseT:
    """
        Loads the module from ``path``.
        Passes all ``*args`` to the module, during loading.
        Raises ``ModuleError`` if a module is not found at ``path``.

        For more information see https://redis.io/commands/module-load
        """
    return self.execute_command('MODULE LOAD', path, *args)