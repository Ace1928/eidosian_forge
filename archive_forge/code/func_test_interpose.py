import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_interpose():
    assert 'a' == first(rest(interpose('a', range(1000000000))))
    assert 'tXaXrXzXaXn' == ''.join(interpose('X', 'tarzan'))
    assert list(interpose(0, itertools.repeat(1, 4))) == [1, 0, 1, 0, 1, 0, 1]
    assert list(interpose('.', ['a', 'b', 'c'])) == ['a', '.', 'b', '.', 'c']