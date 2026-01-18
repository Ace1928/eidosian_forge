import sys
import traceback
from bisect import bisect_left, bisect_right, insort
from itertools import chain, repeat, starmap
from math import log
from operator import add, eq, ne, gt, ge, lt, le, iadd
from textwrap import dedent
from functools import wraps
from sys import hexversion
def islice(self, start=None, stop=None, reverse=False):
    """Return an iterator that slices sorted list from `start` to `stop`.

        The `start` and `stop` index are treated inclusive and exclusive,
        respectively.

        Both `start` and `stop` default to `None` which is automatically
        inclusive of the beginning and end of the sorted list.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> sl = SortedList('abcdefghij')
        >>> it = sl.islice(2, 6)
        >>> list(it)
        ['c', 'd', 'e', 'f']

        :param int start: start index (inclusive)
        :param int stop: stop index (exclusive)
        :param bool reverse: yield values in reverse order
        :return: iterator

        """
    _len = self._len
    if not _len:
        return iter(())
    start, stop, _ = slice(start, stop).indices(self._len)
    if start >= stop:
        return iter(())
    _pos = self._pos
    min_pos, min_idx = _pos(start)
    if stop == _len:
        max_pos = len(self._lists) - 1
        max_idx = len(self._lists[-1])
    else:
        max_pos, max_idx = _pos(stop)
    return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)