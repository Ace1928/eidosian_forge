import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_dtypes(using_infer_string):
    idx_multitype = MultiIndex.from_product([[1, 2, 3], ['a', 'b', 'c'], pd.date_range('20200101', periods=2, tz='UTC')], names=['int', 'string', 'dt'])
    exp = 'object' if not using_infer_string else 'string'
    expected = pd.Series({'int': np.dtype('int64'), 'string': exp, 'dt': DatetimeTZDtype(tz='utc')})
    tm.assert_series_equal(expected, idx_multitype.dtypes)