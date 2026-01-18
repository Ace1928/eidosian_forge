import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
@pytest.mark.parametrize('offset', [-5, 5])
def test_indexing_over_hashtable_size_cutoff(self, monkeypatch, offset):
    size_cutoff = 20
    n = size_cutoff + offset
    with monkeypatch.context():
        monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
        s = Series(np.arange(n), MultiIndex.from_arrays((['a'] * n, np.arange(n))))
        assert s['a', 5] == 5
        assert s['a', 6] == 6
        assert s['a', 7] == 7