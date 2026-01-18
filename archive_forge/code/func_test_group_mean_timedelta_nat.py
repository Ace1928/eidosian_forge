import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_group_mean_timedelta_nat():
    data = Series(['1 day', '3 days', 'NaT'], dtype='timedelta64[ns]')
    expected = Series(['2 days'], dtype='timedelta64[ns]', index=np.array([0]))
    result = data.groupby([0, 0, 0]).mean()
    tm.assert_series_equal(result, expected)