from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('df, expected', [(np.array([1, 2], dtype=object), np.array([-1, -2], dtype=object)), ([Decimal('1.0'), Decimal('2.0')], [Decimal('-1.0'), Decimal('-2.0')])])
def test_neg_object(self, df, expected):
    df = pd.DataFrame({'a': df})
    expected = pd.DataFrame({'a': expected})
    tm.assert_frame_equal(-df, expected)
    tm.assert_series_equal(-df['a'], expected['a'])