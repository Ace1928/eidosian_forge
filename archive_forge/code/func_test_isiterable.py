import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_isiterable():

    class IterIterable:

        def __iter__(self):
            return iter(['a', 'b', 'c'])

    class GetItemIterable:

        def __getitem__(self, item):
            return ['a', 'b', 'c'][item]

    class NotIterable:
        __iter__ = None

    class NotIterableEvenWithGetItem:
        __iter__ = None

        def __getitem__(self, item):
            return ['a', 'b', 'c'][item]
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(IterIterable()) is True
    assert isiterable(GetItemIterable()) is True
    assert isiterable(5) is False
    assert isiterable(NotIterable()) is False
    assert isiterable(NotIterableEvenWithGetItem()) is False