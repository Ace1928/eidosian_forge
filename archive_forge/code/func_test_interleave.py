import itertools
from itertools import starmap
from toolz.utils import raises
from functools import partial
from random import Random
from pickle import dumps, loads
from toolz.itertoolz import (remove, groupby, merge_sorted,
from operator import add, mul
def test_interleave():
    assert ''.join(interleave(('ABC', '123'))) == 'A1B2C3'
    assert ''.join(interleave(('ABC', '1'))) == 'A1BC'