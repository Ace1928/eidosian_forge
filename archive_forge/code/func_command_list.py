import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def command_list(self, module: Optional[str]=None, category: Optional[str]=None, pattern: Optional[str]=None) -> ResponseT:
    """
        Return an array of the server's command names.
        You can use one of the following filters:
        ``module``: get the commands that belong to the module
        ``category``: get the commands in the ACL category
        ``pattern``: get the commands that match the given pattern

        For more information see https://redis.io/commands/command-list/
        """
    pieces = []
    if module is not None:
        pieces.extend(['MODULE', module])
    if category is not None:
        pieces.extend(['ACLCAT', category])
    if pattern is not None:
        pieces.extend(['PATTERN', pattern])
    if pieces:
        pieces.insert(0, 'FILTERBY')
    return self.execute_command('COMMAND LIST', *pieces)