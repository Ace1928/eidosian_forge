import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct('Hello') is False
    assert isdistinct('World') is True
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False