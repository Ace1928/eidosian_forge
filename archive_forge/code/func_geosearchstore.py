import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def geosearchstore(self, dest: KeyT, name: KeyT, member: Union[FieldT, None]=None, longitude: Union[float, None]=None, latitude: Union[float, None]=None, unit: str='m', radius: Union[float, None]=None, width: Union[float, None]=None, height: Union[float, None]=None, sort: Union[str, None]=None, count: Union[int, None]=None, any: bool=False, storedist: bool=False) -> ResponseT:
    """
        This command is like GEOSEARCH, but stores the result in
        ``dest``. By default, it stores the results in the destination
        sorted set with their geospatial information.
        if ``store_dist`` set to True, the command will stores the
        items in a sorted set populated with their distance from the
        center of the circle or box, as a floating-point number.

        For more information see https://redis.io/commands/geosearchstore
        """
    return self._geosearchgeneric('GEOSEARCHSTORE', dest, name, member=member, longitude=longitude, latitude=latitude, unit=unit, radius=radius, width=width, height=height, sort=sort, count=count, any=any, withcoord=None, withdist=None, withhash=None, store=None, store_dist=storedist)