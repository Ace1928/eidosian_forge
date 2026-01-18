from __future__ import annotations
from .base import *
def get_lookup_key(self, key: Optional[str]=None) -> str:
    """
        Returns the lookup key for the given key
        """
    if key is None:
        return self.name_lookup_key
    key = str(key)
    return f'{self.name_lookup_key}:{key}' if self.name_prefix_enabled and self.name_lookup_key not in key else key