from io import StringIO
from string import ascii_uppercase as uppercase
import textwrap
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
def test_info_memory_usage_bug_on_multiindex():
    N = 100
    M = len(uppercase)
    index = MultiIndex.from_product([list(uppercase), date_range('20160101', periods=N)], names=['id', 'date'])
    s = Series(np.random.randn(N * M), index=index)
    unstacked = s.unstack('id')
    assert s.values.nbytes == unstacked.values.nbytes
    assert s.memory_usage(deep=True) > unstacked.memory_usage(deep=True).sum()
    diff = unstacked.memory_usage(deep=True).sum() - s.memory_usage(deep=True)
    assert diff < 2000