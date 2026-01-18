import inspect
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [Series, pd.Index, pd.array])
def test_dt64_mean(self, tz_naive_fixture, box):
    tz = tz_naive_fixture
    dti = date_range('2001-01-01', periods=11, tz=tz)
    dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
    dtarr = dti._data
    obj = box(dtarr)
    assert obj.mean() == pd.Timestamp('2001-01-06', tz=tz)
    assert obj.mean(skipna=False) == pd.Timestamp('2001-01-06', tz=tz)
    dtarr[-2] = pd.NaT
    obj = box(dtarr)
    assert obj.mean() == pd.Timestamp('2001-01-06 07:12:00', tz=tz)
    assert obj.mean(skipna=False) is pd.NaT