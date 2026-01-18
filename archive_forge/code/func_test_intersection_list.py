from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_intersection_list(self):
    values = [Timestamp('2020-01-01'), Timestamp('2020-02-01')]
    idx = DatetimeIndex(values, name='a')
    res = idx.intersection(values)
    tm.assert_index_equal(res, idx)