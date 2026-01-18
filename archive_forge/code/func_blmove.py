import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def blmove(self, first_list: str, second_list: str, timeout: int, src: str='LEFT', dest: str='RIGHT') -> ResponseT:
    """
        Blocking version of lmove.

        For more information see https://redis.io/commands/blmove
        """
    params = [first_list, second_list, src, dest, timeout]
    return self.execute_command('BLMOVE', *params)