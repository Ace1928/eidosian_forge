import sys
import traceback
from bisect import bisect_left, bisect_right, insort
from itertools import chain, repeat, starmap
from math import log
from operator import add, eq, ne, gt, ge, lt, le, iadd
from textwrap import dedent
from functools import wraps
from sys import hexversion
def irange_key(self, min_key=None, max_key=None, inclusive=(True, True), reverse=False):
    """Create an iterator of values between `min_key` and `max_key`.

        Both `min_key` and `max_key` default to `None` which is automatically
        inclusive of the beginning and end of the sorted-key list.

        The argument `inclusive` is a pair of booleans that indicates whether
        the minimum and maximum ought to be included in the range,
        respectively. The default is ``(True, True)`` such that the range is
        inclusive of both minimum and maximum.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> from operator import neg
        >>> skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)
        >>> it = skl.irange_key(-14, -12)
        >>> list(it)
        [14, 13, 12]

        :param min_key: minimum key to start iterating
        :param max_key: maximum key to stop iterating
        :param inclusive: pair of booleans
        :param bool reverse: yield values in reverse order
        :return: iterator

        """
    _maxes = self._maxes
    if not _maxes:
        return iter(())
    _keys = self._keys
    if min_key is None:
        min_pos = 0
        min_idx = 0
    elif inclusive[0]:
        min_pos = bisect_left(_maxes, min_key)
        if min_pos == len(_maxes):
            return iter(())
        min_idx = bisect_left(_keys[min_pos], min_key)
    else:
        min_pos = bisect_right(_maxes, min_key)
        if min_pos == len(_maxes):
            return iter(())
        min_idx = bisect_right(_keys[min_pos], min_key)
    if max_key is None:
        max_pos = len(_maxes) - 1
        max_idx = len(_keys[max_pos])
    elif inclusive[1]:
        max_pos = bisect_right(_maxes, max_key)
        if max_pos == len(_maxes):
            max_pos -= 1
            max_idx = len(_keys[max_pos])
        else:
            max_idx = bisect_right(_keys[max_pos], max_key)
    else:
        max_pos = bisect_left(_maxes, max_key)
        if max_pos == len(_maxes):
            max_pos -= 1
            max_idx = len(_keys[max_pos])
        else:
            max_idx = bisect_left(_keys[max_pos], max_key)
    return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)