from __future__ import annotations
from collections import abc
from datetime import (
from decimal import Decimal
import operator
import os
from typing import (
from dateutil.tz import (
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
from pytz import (
from pandas._config.config import _get_option
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import (
from pandas.util.version import Version
import zoneinfo
@pytest.fixture
def multiindex_year_month_day_dataframe_random_data():
    """
    DataFrame with 3 level MultiIndex (year, month, day) covering
    first 100 business days from 2000-01-01 with random data
    """
    tdf = DataFrame(np.random.default_rng(2).standard_normal((100, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=100, freq='B'))
    ymd = tdf.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day]).sum()
    ymd.index = ymd.index.set_levels([lev.astype('i8') for lev in ymd.index.levels])
    ymd.index.set_names(['year', 'month', 'day'], inplace=True)
    return ymd