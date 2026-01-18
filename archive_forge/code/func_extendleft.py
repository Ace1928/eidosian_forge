from collections.abc import Sequence, Hashable
from itertools import islice, chain
from numbers import Integral
from typing import TypeVar, Generic
from pyrsistent._plist import plist
def extendleft(self, iterable):
    """
        Return new deque with all elements of iterable appended to the left.

        NB! The elements will be inserted in reverse order compared to the order in the iterable.

        >>> pdeque([1, 2]).extendleft([3, 4])
        pdeque([4, 3, 1, 2])
        """
    new_left_list, new_right_list, extend_count = self._extend(self._left_list, self._right_list, iterable)
    return PDeque(new_left_list, new_right_list, self._length + extend_count, self._maxlen)