import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def renamenx(self, src: KeyT, dst: KeyT):
    """
        Rename key ``src`` to ``dst`` if ``dst`` doesn't already exist

        For more information see https://redis.io/commands/renamenx
        """
    return self.execute_command('RENAMENX', src, dst)