import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
@classmethod
def common_prefix(cls, prefix, key):
    """Given 2 strings, return the longest prefix common to both.

        :param prefix: This has been the common prefix for other keys, so it is
            more likely to be the common prefix in this case as well.
        :param key: Another string to compare to
        """
    if key.startswith(prefix):
        return prefix
    pos = -1
    for pos, (left, right) in enumerate(zip(prefix, key)):
        if left != right:
            pos -= 1
            break
    common = prefix[:pos + 1]
    return common