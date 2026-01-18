import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def georadius(self, name: KeyT, longitude: float, latitude: float, radius: float, unit: Union[str, None]=None, withdist: bool=False, withcoord: bool=False, withhash: bool=False, count: Union[int, None]=None, sort: Union[str, None]=None, store: Union[KeyT, None]=None, store_dist: Union[KeyT, None]=None, any: bool=False) -> ResponseT:
    """
        Return the members of the specified key identified by the
        ``name`` argument which are within the borders of the area specified
        with the ``latitude`` and ``longitude`` location and the maximum
        distance from the center specified by the ``radius`` value.

        The units must be one of the following : m, km mi, ft. By default

        ``withdist`` indicates to return the distances of each place.

        ``withcoord`` indicates to return the latitude and longitude of
        each place.

        ``withhash`` indicates to return the geohash string of each place.

        ``count`` indicates to return the number of elements up to N.

        ``sort`` indicates to return the places in a sorted way, ASC for
        nearest to fairest and DESC for fairest to nearest.

        ``store`` indicates to save the places names in a sorted set named
        with a specific key, each element of the destination sorted set is
        populated with the score got from the original geo sorted set.

        ``store_dist`` indicates to save the places names in a sorted set
        named with a specific key, instead of ``store`` the sorted set
        destination score is set with the distance.

        For more information see https://redis.io/commands/georadius
        """
    return self._georadiusgeneric('GEORADIUS', name, longitude, latitude, radius, unit=unit, withdist=withdist, withcoord=withcoord, withhash=withhash, count=count, sort=sort, store=store, store_dist=store_dist, any=any)