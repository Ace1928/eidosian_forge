from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_get_dependencies_nested():
    dsk = {'x': 1, 'y': 2, 'z': (add, (inc, [['x']]), 'y')}
    assert get_dependencies(dsk, 'z') == {'x', 'y'}
    assert sorted(get_dependencies(dsk, 'z', as_list=True)) == ['x', 'y']