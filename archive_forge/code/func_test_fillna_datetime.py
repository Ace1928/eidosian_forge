import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_datetime(self, datetime_frame):
    tf = datetime_frame
    tf.loc[tf.index[:5], 'A'] = np.nan
    tf.loc[tf.index[-5:], 'A'] = np.nan
    zero_filled = datetime_frame.fillna(0)
    assert (zero_filled.loc[zero_filled.index[:5], 'A'] == 0).all()
    msg = "DataFrame.fillna with 'method' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        padded = datetime_frame.fillna(method='pad')
    assert np.isnan(padded.loc[padded.index[:5], 'A']).all()
    assert (padded.loc[padded.index[-5:], 'A'] == padded.loc[padded.index[-5], 'A']).all()
    msg = "Must specify a fill 'value' or 'method'"
    with pytest.raises(ValueError, match=msg):
        datetime_frame.fillna()
    msg = "Cannot specify both 'value' and 'method'"
    with pytest.raises(ValueError, match=msg):
        datetime_frame.fillna(5, method='ffill')