from collections import defaultdict, Counter
from typing import Dict, Generator, List, Optional, TypeVar
def pop_cached_object(self, key: T) -> Optional[U]:
    """Get one cached object for a key.

        This will remove the object from the cache.

        Args:
            key: Group key.

        Returns:
            Cached object.
        """
    if not self.has_cached_object(key):
        return None
    self._num_cached_objects -= 1
    return self._cached_objects[key].pop(0)