import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_mapcat():
    assert list(mapcat(identity, [[1, 2, 3], [4, 5, 6]])) == [1, 2, 3, 4, 5, 6]
    assert list(mapcat(reversed, [[3, 2, 1, 0], [6, 5, 4], [9, 8, 7]])) == list(range(10))
    inc = lambda i: i + 1
    assert [4, 5, 6, 7, 8, 9] == list(mapcat(partial(map, inc), [[3, 4, 5], [6, 7, 8]]))