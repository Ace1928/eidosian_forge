from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_isin_unsigned_dtype(self):
    ser = Series([1378774140726870442], dtype=np.uint64)
    result = ser.isin([1378774140726870528])
    expected = Series(False)
    tm.assert_series_equal(result, expected)