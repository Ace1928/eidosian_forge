import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_on_series_buglet(self):
    df = DataFrame({'a': [1, 1]})
    ds = Series([2], index=[1], name='b')
    result = df.join(ds, on='a')
    expected = DataFrame({'a': [1, 1], 'b': [2, 2]}, index=df.index)
    tm.assert_frame_equal(result, expected)