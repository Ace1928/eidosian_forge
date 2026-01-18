from __future__ import annotations
from .base import *
def hpop(self, field: str, key: Optional[str]=None) -> Any:
    """
        Returns the value for the given key
        """
    key = self.get_key(key)
    value = self.kdb.hget(key, field)
    self.kdb.hdel(key, field)
    return self.decode(value)