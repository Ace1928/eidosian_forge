from collections.abc import Sequence, Hashable
from itertools import islice, chain
from numbers import Integral
from typing import TypeVar, Generic
from pyrsistent._plist import plist
def pdeque(iterable=(), maxlen=None):
    """
    Return deque containing the elements of iterable. If maxlen is specified then
    len(iterable) - maxlen elements are discarded from the left to if len(iterable) > maxlen.

    >>> pdeque([1, 2, 3])
    pdeque([1, 2, 3])
    >>> pdeque([1, 2, 3, 4], maxlen=2)
    pdeque([3, 4], maxlen=2)
    """
    t = tuple(iterable)
    if maxlen is not None:
        t = t[-maxlen:]
    length = len(t)
    pivot = int(length / 2)
    left = plist(t[:pivot])
    right = plist(t[pivot:], reverse=True)
    return PDeque(left, right, length, maxlen)