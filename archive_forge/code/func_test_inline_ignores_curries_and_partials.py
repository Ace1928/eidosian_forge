from __future__ import annotations
import itertools
import pickle
from functools import partial
import pytest
import dask
from dask.base import tokenize
from dask.core import get_dependencies
from dask.local import get_sync
from dask.optimization import (
from dask.utils import apply, partial_by_order
from dask.utils_test import add, inc
def test_inline_ignores_curries_and_partials():
    dsk = {'x': 1, 'y': 2, 'a': (partial(add, 1), 'x'), 'b': (inc, 'a')}
    result = inline_functions(dsk, [], fast_functions={add})
    assert result['b'] == (inc, dsk['a'])
    assert 'a' not in result