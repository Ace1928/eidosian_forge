import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
@classmethod
def common_prefix_for_keys(cls, keys):
    """Given a list of keys, find their common prefix.

        :param keys: An iterable of strings.
        :return: The longest common prefix of all keys.
        """
    common_prefix = None
    for key in keys:
        if common_prefix is None:
            common_prefix = key
            continue
        common_prefix = cls.common_prefix(common_prefix, key)
        if not common_prefix:
            return b''
    return common_prefix