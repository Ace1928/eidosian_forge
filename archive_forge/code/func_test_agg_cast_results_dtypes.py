import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_cast_results_dtypes():
    u = [dt.datetime(2015, x + 1, 1) for x in range(12)]
    v = list('aaabbbbbbccd')
    df = DataFrame({'X': v, 'Y': u})
    result = df.groupby('X')['Y'].agg(len)
    expected = df.groupby('X')['Y'].count()
    tm.assert_series_equal(result, expected)