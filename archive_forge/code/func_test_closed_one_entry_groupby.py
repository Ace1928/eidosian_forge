from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('func', ['min', 'max'])
def test_closed_one_entry_groupby(func):
    ser = DataFrame(data={'A': [1, 1, 2], 'B': [3, 2, 1]}, index=date_range('2000', periods=3))
    result = getattr(ser.groupby('A', sort=False)['B'].rolling('10D', closed='left'), func)()
    exp_idx = MultiIndex.from_arrays(arrays=[[1, 1, 2], ser.index], names=('A', None))
    expected = Series(data=[np.nan, 3, np.nan], index=exp_idx, name='B')
    tm.assert_series_equal(result, expected)