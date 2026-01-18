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
def test_getitem_1tuple_slice_without_multiindex():
    ser = Series(range(5))
    key = (slice(3),)
    result = ser[key]
    expected = ser[key[0]]
    tm.assert_series_equal(result, expected)