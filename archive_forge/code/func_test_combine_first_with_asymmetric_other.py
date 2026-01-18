from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [1, 1.0])
def test_combine_first_with_asymmetric_other(self, val):
    df1 = DataFrame({'isNum': [val]})
    df2 = DataFrame({'isBool': [True]})
    res = df1.combine_first(df2)
    exp = DataFrame({'isBool': [True], 'isNum': [val]})
    tm.assert_frame_equal(res, exp)