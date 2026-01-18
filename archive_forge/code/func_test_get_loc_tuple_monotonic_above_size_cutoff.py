import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.slow
def test_get_loc_tuple_monotonic_above_size_cutoff(self, monkeypatch):
    with monkeypatch.context():
        monkeypatch.setattr(libindex, '_SIZE_CUTOFF', 100)
        lev = list('ABCD')
        dti = pd.date_range('2016-01-01', periods=10)
        mi = pd.MultiIndex.from_product([lev, range(5), dti])
        oidx = mi.to_flat_index()
        loc = len(oidx) // 2
        tup = oidx[loc]
        res = oidx.get_loc(tup)
    assert res == loc