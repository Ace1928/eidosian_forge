from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_loc_assignment_slice_state(self):
    df = DataFrame({'a': [10, 20, 30]})
    with tm.raises_chained_assignment_error():
        df['a'].loc[4] = 40
    tm.assert_frame_equal(df, DataFrame({'a': [10, 20, 30]}))
    tm.assert_series_equal(df['a'], Series([10, 20, 30], name='a'))