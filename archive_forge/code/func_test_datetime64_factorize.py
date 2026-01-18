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
def test_datetime64_factorize(self, writable):
    data = np.array([np.datetime64('2020-01-01T00:00:00.000')], dtype='M8[ns]')
    data.setflags(write=writable)
    expected_codes = np.array([0], dtype=np.intp)
    expected_uniques = np.array(['2020-01-01T00:00:00.000000000'], dtype='datetime64[ns]')
    codes, uniques = pd.factorize(data)
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_numpy_array_equal(uniques, expected_uniques)