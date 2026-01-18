import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def set_maximum_size(self, new_size):
    """Set the size threshold for nodes.

        :param new_size: The size at which no data is added to a node. 0 for
            unlimited.
        """
    self._maximum_size = new_size