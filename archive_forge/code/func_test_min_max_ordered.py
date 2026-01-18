import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
def test_min_max_ordered(self, index_or_series_or_array):
    cat = Categorical(['a', 'b', 'c', 'd'], ordered=True)
    obj = index_or_series_or_array(cat)
    _min = obj.min()
    _max = obj.max()
    assert _min == 'a'
    assert _max == 'd'
    assert np.minimum.reduce(obj) == 'a'
    assert np.maximum.reduce(obj) == 'd'
    cat = Categorical(['a', 'b', 'c', 'd'], categories=['d', 'c', 'b', 'a'], ordered=True)
    obj = index_or_series_or_array(cat)
    _min = obj.min()
    _max = obj.max()
    assert _min == 'd'
    assert _max == 'a'
    assert np.minimum.reduce(obj) == 'd'
    assert np.maximum.reduce(obj) == 'a'