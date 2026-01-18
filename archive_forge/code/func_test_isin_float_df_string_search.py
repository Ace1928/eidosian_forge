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
def test_isin_float_df_string_search(self):
    """Comparing df with floats (1.4245,2.32441) with a string at isin() ("1.4245")
        -> should not match values because float 1.4245 is not equal str 1.4245"""
    df = DataFrame({'values': [1.4245, 2.32441]})
    result = df.isin(np.array(['1.4245'], dtype=object))
    expected_false = DataFrame({'values': [False, False]})
    tm.assert_frame_equal(result, expected_false)