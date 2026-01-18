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
def test_getitem_median_slice_bug(self):
    index = date_range('20090415', '20090519', freq='2B')
    ser = Series(np.random.default_rng(2).standard_normal(13), index=index)
    indexer = [slice(6, 7, None)]
    msg = 'Indexing with a single-item list'
    with pytest.raises(ValueError, match=msg):
        ser[indexer]
    result = ser[indexer[0],]
    expected = ser[indexer[0]]
    tm.assert_series_equal(result, expected)