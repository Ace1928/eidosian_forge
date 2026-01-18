import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import Timestamp
def test_datetime_timedelta_quantiles(self):
    assert pd.isna(Series([], dtype='M8[ns]').quantile(0.5))
    assert pd.isna(Series([], dtype='m8[ns]').quantile(0.5))