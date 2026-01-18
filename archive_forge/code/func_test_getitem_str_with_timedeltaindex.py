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
def test_getitem_str_with_timedeltaindex(self):
    rng = timedelta_range('1 day 10:11:12', freq='h', periods=500)
    ser = Series(np.arange(len(rng)), index=rng)
    key = '6 days, 23:11:12'
    indexer = rng.get_loc(key)
    assert indexer == 133
    result = ser[key]
    assert result == ser.iloc[133]
    msg = "^Timedelta\\('50 days 00:00:00'\\)$"
    with pytest.raises(KeyError, match=msg):
        rng.get_loc('50 days')
    with pytest.raises(KeyError, match=msg):
        ser['50 days']