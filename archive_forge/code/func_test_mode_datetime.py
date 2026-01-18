from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dropna, expected1, expected2', [(True, ['1900-05-03', '2011-01-03', '2013-01-02'], ['2011-01-03', '2013-01-02']), (False, [np.nan], [np.nan, '2011-01-03', '2013-01-02'])])
def test_mode_datetime(self, dropna, expected1, expected2):
    s = Series(['2011-01-03', '2013-01-02', '1900-05-03', 'nan', 'nan'], dtype='M8[ns]')
    result = s.mode(dropna)
    expected1 = Series(expected1, dtype='M8[ns]')
    tm.assert_series_equal(result, expected1)
    s = Series(['2011-01-03', '2013-01-02', '1900-05-03', '2011-01-03', '2013-01-02', 'nan', 'nan'], dtype='M8[ns]')
    result = s.mode(dropna)
    expected2 = Series(expected2, dtype='M8[ns]')
    tm.assert_series_equal(result, expected2)