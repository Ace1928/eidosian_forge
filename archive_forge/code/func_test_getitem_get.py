import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_get(string_series, object_series):
    msg = 'Series.__getitem__ treating keys as positions is deprecated'
    for obj in [string_series, object_series]:
        idx = obj.index[5]
        assert obj[idx] == obj.get(idx)
        assert obj[idx] == obj.iloc[5]
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert string_series.get(-1) == string_series.get(string_series.index[-1])
    assert string_series.iloc[5] == string_series.get(string_series.index[5])