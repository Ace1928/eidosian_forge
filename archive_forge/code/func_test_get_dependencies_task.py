from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_get_dependencies_task():
    dsk = {'x': 1, 'y': 2, 'z': ['x', [(inc, 'y')]]}
    assert get_dependencies(dsk, task=(inc, 'x')) == {'x'}
    assert get_dependencies(dsk, task=(inc, 'x'), as_list=True) == ['x']