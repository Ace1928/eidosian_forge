import functools
from typing import Optional
from .base import VariableTracker
def is_realized(self):
    return self._cache.vt is not None