import collections
from typing import Any, Set
import weakref
def intersection(self, items):
    return self._storage.intersection([self._wrap_key(item) for item in items])