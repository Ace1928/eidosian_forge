import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unwrap', [True, False])
def test_contains_na_dtype(self, unwrap):
    dti = pd.date_range('2016-01-01', periods=100).insert(0, pd.NaT)
    pi = dti.to_period('D')
    tdi = dti - dti[-1]
    ci = CategoricalIndex(dti)
    obj = ci
    if unwrap:
        obj = ci._data
    assert np.nan in obj
    assert None in obj
    assert pd.NaT in obj
    assert np.datetime64('NaT') in obj
    assert np.timedelta64('NaT') not in obj
    obj2 = CategoricalIndex(tdi)
    if unwrap:
        obj2 = obj2._data
    assert np.nan in obj2
    assert None in obj2
    assert pd.NaT in obj2
    assert np.datetime64('NaT') not in obj2
    assert np.timedelta64('NaT') in obj2
    obj3 = CategoricalIndex(pi)
    if unwrap:
        obj3 = obj3._data
    assert np.nan in obj3
    assert None in obj3
    assert pd.NaT in obj3
    assert np.datetime64('NaT') not in obj3
    assert np.timedelta64('NaT') not in obj3