from collections.abc import Sequence, Hashable
from numbers import Integral
from functools import reduce
from typing import Generic, TypeVar
def mcons(self, iterable):
    """
        Return a new list with all elements of iterable repeatedly cons:ed to the current list.
        NB! The elements will be inserted in the reverse order of the iterable.
        Runs in O(len(iterable)).

        >>> plist([1, 2]).mcons([3, 4])
        plist([4, 3, 1, 2])
        """
    head = self
    for elem in iterable:
        head = head.cons(elem)
    return head