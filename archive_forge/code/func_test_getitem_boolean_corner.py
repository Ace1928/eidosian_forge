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
def test_getitem_boolean_corner(self, datetime_series):
    ts = datetime_series
    mask_shifted = ts.shift(1, freq=BDay()) > ts.median()
    msg = 'Unalignable boolean Series provided as indexer \\(index of the boolean Series and of the indexed object do not match'
    with pytest.raises(IndexingError, match=msg):
        ts[mask_shifted]
    with pytest.raises(IndexingError, match=msg):
        ts.loc[mask_shifted]