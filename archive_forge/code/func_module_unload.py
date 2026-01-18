import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def module_unload(self, name) -> ResponseT:
    """
        Unloads the module ``name``.
        Raises ``ModuleError`` if ``name`` is not in loaded modules.

        For more information see https://redis.io/commands/module-unload
        """
    return self.execute_command('MODULE UNLOAD', name)