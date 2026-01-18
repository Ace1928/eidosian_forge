import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
@property
def currsize(self):
    with self.__timer as time:
        self.expire(time)
        return super().currsize