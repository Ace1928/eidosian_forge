import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_accumulate_works_on_consumable_iterables():
    assert list(accumulate(add, iter((1, 2, 3)))) == [1, 3, 6]