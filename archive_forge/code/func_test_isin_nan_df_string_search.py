from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_isin_nan_df_string_search(self):
    """Comparing df with nan value (np.nan,2) with a string at isin() ("NaN")
        -> should not match values because np.nan is not equal str NaN"""
    df = DataFrame({'values': [np.nan, 2]})
    result = df.isin(np.array(['NaN'], dtype=object))
    expected_false = DataFrame({'values': [False, False]})
    tm.assert_frame_equal(result, expected_false)