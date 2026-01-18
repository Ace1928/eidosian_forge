import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def function_restore(self, payload: str, policy: Optional[str]='APPEND') -> Union[Awaitable[str], str]:
    """
        Restore libraries from the serialized ``payload``.
        You can use the optional policy argument to provide a policy
        for handling existing libraries.

        For more information see https://redis.io/commands/function-restore
        """
    return self.execute_command('FUNCTION RESTORE', payload, policy)