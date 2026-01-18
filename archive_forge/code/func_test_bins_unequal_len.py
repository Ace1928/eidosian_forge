from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_bins_unequal_len():
    series = Series([np.nan, np.nan, 1, 1, 2, 2, 3, 3, 4, 4])
    bins = pd.cut(series.dropna().values, 4)
    with pytest.raises(ValueError, match='Grouper and axis must be same length'):
        series.groupby(bins).mean()