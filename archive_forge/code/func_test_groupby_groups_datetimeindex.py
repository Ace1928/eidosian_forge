from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_groups_datetimeindex(self):
    periods = 1000
    ind = date_range(start='2012/1/1', freq='5min', periods=periods)
    df = DataFrame({'high': np.arange(periods), 'low': np.arange(periods)}, index=ind)
    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    groups = grouped.groups
    assert isinstance(next(iter(groups.keys())), datetime)