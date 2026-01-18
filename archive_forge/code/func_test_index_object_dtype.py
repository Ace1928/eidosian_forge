import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('values_constructor', [list, np.array, PeriodIndex, PeriodArray._from_sequence])
def test_index_object_dtype(self, values_constructor):
    periods = [Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')]
    values = values_constructor(periods)
    result = Index(values, dtype=object)
    assert type(result) is Index
    tm.assert_numpy_array_equal(result.values, np.array(values))