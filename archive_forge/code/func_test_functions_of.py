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
def test_functions_of():
    a = lambda x: x
    b = lambda x: x
    assert functions_of((a, 1)) == {a}
    assert functions_of((a, (b, 1))) == {a, b}
    assert functions_of((a, [(b, 1)])) == {a, b}
    assert functions_of((a, [[[(b, 1)]]])) == {a, b}
    assert functions_of(1) == set()
    assert functions_of(a) == set()
    assert functions_of((a,)) == {a}