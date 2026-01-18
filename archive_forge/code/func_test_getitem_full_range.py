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
def test_getitem_full_range(self):
    ser = Series(range(5), index=list(range(5)))
    result = ser[list(range(5))]
    tm.assert_series_equal(result, ser)