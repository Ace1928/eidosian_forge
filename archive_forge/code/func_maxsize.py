import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
@property
def maxsize(self):
    """The maximum size of the cache."""
    return self.__maxsize