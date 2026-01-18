import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def sintercard(self, numkeys: int, keys: List[str], limit: int=0) -> Union[Awaitable[int], int]:
    """
        Return the cardinality of the intersect of multiple sets specified by ``keys`.

        When LIMIT provided (defaults to 0 and means unlimited), if the intersection
        cardinality reaches limit partway through the computation, the algorithm will
        exit and yield limit as the cardinality

        For more information see https://redis.io/commands/sintercard
        """
    args = [numkeys, *keys, 'LIMIT', limit]
    return self.execute_command('SINTERCARD', *args)