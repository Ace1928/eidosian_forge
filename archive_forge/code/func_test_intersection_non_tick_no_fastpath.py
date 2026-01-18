from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_intersection_non_tick_no_fastpath(self):
    dti = DatetimeIndex(['2018-12-31', '2019-03-31', '2019-06-30', '2019-09-30', '2019-12-31', '2020-03-31'], freq='QE-DEC')
    result = dti[::2].intersection(dti[1::2])
    expected = dti[:0]
    tm.assert_index_equal(result, expected)