import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_groupby_non_callable():
    assert groupby(0, [(1, 2), (1, 3), (2, 2), (2, 4)]) == {1: [(1, 2), (1, 3)], 2: [(2, 2), (2, 4)]}
    assert groupby([0], [(1, 2), (1, 3), (2, 2), (2, 4)]) == {(1,): [(1, 2), (1, 3)], (2,): [(2, 2), (2, 4)]}
    assert groupby([0, 0], [(1, 2), (1, 3), (2, 2), (2, 4)]) == {(1, 1): [(1, 2), (1, 3)], (2, 2): [(2, 2), (2, 4)]}