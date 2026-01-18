import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_consistency():

    def P1(a):
        return np.percentile(a.dropna(), q=1)
    df = DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 25, 26, 31], 'date': [dt.date(2013, 2, 10), dt.date(2013, 2, 10), dt.date(2013, 2, 11), dt.date(2013, 2, 11)]})
    g = df.groupby('date')
    expected = g.agg([P1])
    expected.columns = expected.columns.levels[0]
    result = g.agg(P1)
    tm.assert_frame_equal(result, expected)