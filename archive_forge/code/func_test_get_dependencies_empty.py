from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_get_dependencies_empty():
    dsk = {'x': (inc,)}
    assert get_dependencies(dsk, 'x') == set()
    assert get_dependencies(dsk, 'x', as_list=True) == []