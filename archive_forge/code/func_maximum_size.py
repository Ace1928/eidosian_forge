import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
@property
def maximum_size(self):
    """What is the upper limit for adding references to a node."""
    return self._maximum_size