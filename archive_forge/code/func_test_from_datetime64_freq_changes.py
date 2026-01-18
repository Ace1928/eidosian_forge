import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import IncompatibleFrequency
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_from_datetime64_freq_changes():
    arr = pd.date_range('2017', periods=3, freq='D')
    result = PeriodArray._from_datetime64(arr, freq='M')
    expected = period_array(['2017-01-01', '2017-01-01', '2017-01-01'], freq='M')
    tm.assert_period_array_equal(result, expected)