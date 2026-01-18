from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_iskey():

    class C:
        pass
    assert iskey('x')
    assert iskey(1)
    assert not iskey(C())
    assert not iskey((C(),))
    assert iskey((1, 2))
    assert iskey(())
    assert iskey(('x',))
    assert not iskey([1, 2])
    assert not iskey({1, 2})
    assert not iskey({1: 2})