from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_int64(self, datetime_series):
    idx = np.int64(5)
    msg = 'Series.__getitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = datetime_series[idx]
    assert res == datetime_series.iloc[5]