import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def function_list(self, library: Optional[str]='*', withcode: Optional[bool]=False) -> Union[Awaitable[List], List]:
    """
        Return information about the functions and libraries.
        :param library: pecify a pattern for matching library names
        :param withcode: cause the server to include the libraries source
         implementation in the reply
        """
    args = ['LIBRARYNAME', library]
    if withcode:
        args.append('WITHCODE')
    return self.execute_command('FUNCTION LIST', *args)