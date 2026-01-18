import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def sort_ro(self, key: str, start: Optional[int]=None, num: Optional[int]=None, by: Optional[str]=None, get: Optional[List[str]]=None, desc: bool=False, alpha: bool=False) -> list:
    """
        Returns the elements contained in the list, set or sorted set at key.
        (read-only variant of the SORT command)

        ``start`` and ``num`` allow for paging through the sorted data

        ``by`` allows using an external key to weight and sort the items.
            Use an "*" to indicate where in the key the item value is located

        ``get`` allows for returning items from external keys rather than the
            sorted data itself.  Use an "*" to indicate where in the key
            the item value is located

        ``desc`` allows for reversing the sort

        ``alpha`` allows for sorting lexicographically rather than numerically

        For more information see https://redis.io/commands/sort_ro
        """
    return self.sort(key, start=start, num=num, by=by, get=get, desc=desc, alpha=alpha)