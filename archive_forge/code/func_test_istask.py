from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_istask():
    assert istask((inc, 1))
    assert not istask(1)
    assert not istask((1, 2))
    f = namedtuple('f', ['x', 'y'])
    assert not istask(f(sum, 2))