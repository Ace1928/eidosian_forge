import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_indexer_caching(monkeypatch):
    size_cutoff = 20
    with monkeypatch.context():
        monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
        index = MultiIndex.from_arrays([np.arange(size_cutoff), np.arange(size_cutoff)])
        s = Series(np.zeros(size_cutoff), index=index)
        s[s == 0] = 1
    expected = Series(np.ones(size_cutoff), index=index)
    tm.assert_series_equal(s, expected)