from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_on_int_array(self):
    df = DataFrame({'A': Series([1, 2, np.nan], dtype='Int64'), 'B': 1})
    result = merge(df, df, on='A')
    expected = DataFrame({'A': Series([1, 2, np.nan], dtype='Int64'), 'B_x': 1, 'B_y': 1})
    tm.assert_frame_equal(result, expected)