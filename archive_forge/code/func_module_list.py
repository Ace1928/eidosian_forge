import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def module_list(self) -> ResponseT:
    """
        Returns a list of dictionaries containing the name and version of
        all loaded modules.

        For more information see https://redis.io/commands/module-list
        """
    return self.execute_command('MODULE LIST')