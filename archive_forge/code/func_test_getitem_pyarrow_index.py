import re
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_pyarrow_index(self, frame_or_series):
    pytest.importorskip('pyarrow')
    obj = frame_or_series(range(5), index=date_range('2020', freq='D', periods=5).astype('timestamp[us][pyarrow]'))
    result = obj.loc[obj.index[:-3]]
    expected = frame_or_series(range(2), index=date_range('2020', freq='D', periods=2).astype('timestamp[us][pyarrow]'))
    tm.assert_equal(result, expected)