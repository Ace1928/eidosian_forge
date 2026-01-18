import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def georadiusbymember(self, name: KeyT, member: FieldT, radius: float, unit: Union[str, None]=None, withdist: bool=False, withcoord: bool=False, withhash: bool=False, count: Union[int, None]=None, sort: Union[str, None]=None, store: Union[KeyT, None]=None, store_dist: Union[KeyT, None]=None, any: bool=False) -> ResponseT:
    """
        This command is exactly like ``georadius`` with the sole difference
        that instead of taking, as the center of the area to query, a longitude
        and latitude value, it takes the name of a member already existing
        inside the geospatial index represented by the sorted set.

        For more information see https://redis.io/commands/georadiusbymember
        """
    return self._georadiusgeneric('GEORADIUSBYMEMBER', name, member, radius, unit=unit, withdist=withdist, withcoord=withcoord, withhash=withhash, count=count, sort=sort, store=store, store_dist=store_dist, any=any)