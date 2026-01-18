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
@pytest.mark.parametrize('box', [list, np.array, Index])
def test_getitem_intlist_intervalindex_non_int(self, box):
    dti = date_range('2000-01-03', periods=3)._with_freq(None)
    ii = pd.IntervalIndex.from_breaks(dti)
    ser = Series(range(len(ii)), index=ii)
    expected = ser.iloc[:1]
    key = box([0])
    msg = 'Series.__getitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser[key]
    tm.assert_series_equal(result, expected)