from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_ishashable():

    class C:
        pass
    assert ishashable('x')
    assert ishashable(1)
    assert ishashable(C())
    assert ishashable((1, 2))
    assert not ishashable([1, 2])
    assert not ishashable({1: 2})