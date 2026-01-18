from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('how', ['any', 'all'])
@pytest.mark.parametrize('index,expected', [(DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03']), DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'])), (DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', pd.NaT]), DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'])), (TimedeltaIndex(['1 days', '2 days', '3 days']), TimedeltaIndex(['1 days', '2 days', '3 days'])), (TimedeltaIndex([pd.NaT, '1 days', '2 days', '3 days', pd.NaT]), TimedeltaIndex(['1 days', '2 days', '3 days'])), (PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'), PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M')), (PeriodIndex(['2012-02', '2012-04', 'NaT', '2012-05'], freq='M'), PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'))])
def test_dropna_dt_like(self, how, index, expected):
    result = index.dropna(how=how)
    tm.assert_index_equal(result, expected)