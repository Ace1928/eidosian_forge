from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_with_unsorted_index(self, frame_or_series):
    index = to_datetime(['2021-01-04', '2021-01-02', '2021-01-03', '2021-01-01'])
    result = frame_or_series(range(4), index=index)
    expected = result.reindex(sorted(index))
    expected.index = expected.index._with_freq('infer')
    result = result.asfreq('D')
    tm.assert_equal(result, expected)