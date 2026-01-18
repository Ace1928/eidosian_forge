import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
@staticmethod
def getsizeof(value):
    """Return the size of a cache element's value."""
    return 1