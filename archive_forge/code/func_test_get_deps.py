from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_get_deps():
    dsk = {'a': [1, 2, 3], 'b': 'a', 'c': [1, (inc, 1)], 'd': [(sum, 'c')], 'e': ['b', 'zzz', 'b'], 'f': [['a', 'b'], 2, 3]}
    dependencies, dependents = get_deps(dsk)
    assert dependencies == {'a': set(), 'b': {'a'}, 'c': set(), 'd': {'c'}, 'e': {'b'}, 'f': {'a', 'b'}}
    assert dependents == {'a': {'b', 'f'}, 'b': {'e', 'f'}, 'c': {'d'}, 'd': set(), 'e': set(), 'f': set()}