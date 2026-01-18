import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('s', [pd.Series([1, 2, 3]), pd.Series(pd.date_range('2019', periods=3, tz='UTC'))])
def test_non_object_dtype(s):
    result = s.explode()
    tm.assert_series_equal(result, s)