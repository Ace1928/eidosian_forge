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
@pytest.mark.parametrize('dtype', [np.int64, np.float64, np.uint64])
def test_getitem_intlist_multiindex_numeric_level(self, dtype, box):
    idx = Index(range(4)).astype(dtype)
    dti = date_range('2000-01-03', periods=3)
    mi = pd.MultiIndex.from_product([idx, dti])
    ser = Series(range(len(mi))[::-1], index=mi)
    key = box([5])
    with pytest.raises(KeyError, match='5'):
        ser[key]