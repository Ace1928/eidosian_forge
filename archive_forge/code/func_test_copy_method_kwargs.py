from copy import (
import pytest
from pandas import MultiIndex
import pandas._testing as tm
@pytest.mark.parametrize('deep', [True, False])
@pytest.mark.parametrize('kwarg, value', [('names', ['third', 'fourth'])])
def test_copy_method_kwargs(deep, kwarg, value):
    idx = MultiIndex(levels=[['foo', 'bar'], ['fizz', 'buzz']], codes=[[0, 0, 0, 1], [0, 0, 1, 1]], names=['first', 'second'])
    idx_copy = idx.copy(**{kwarg: value, 'deep': deep})
    assert getattr(idx_copy, kwarg) == value