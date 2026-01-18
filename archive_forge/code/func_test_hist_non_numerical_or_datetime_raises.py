import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_non_numerical_or_datetime_raises(self):
    df = DataFrame({'a': np.random.default_rng(2).random(10), 'b': np.random.default_rng(2).integers(0, 10, 10), 'c': to_datetime(np.random.default_rng(2).integers(1582800000000000000, 1583500000000000000, 10, dtype=np.int64)), 'd': to_datetime(np.random.default_rng(2).integers(1582800000000000000, 1583500000000000000, 10, dtype=np.int64), utc=True)})
    df_o = df.astype(object)
    msg = 'hist method requires numerical or datetime columns, nothing to plot.'
    with pytest.raises(ValueError, match=msg):
        df_o.hist()