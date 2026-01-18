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
@pytest.mark.parametrize('data, expected_codes, expected_uniques', [([1, None, 1, 2], np.array([0, 1, 0, 2], dtype=np.dtype('intp')), np.array([1, np.nan, 2], dtype='O')), ([1, np.nan, 1, 2], np.array([0, 1, 0, 2], dtype=np.dtype('intp')), np.array([1, np.nan, 2], dtype=np.float64))])
def test_int_factorize_use_na_sentinel_false(self, data, expected_codes, expected_uniques):
    msg = 'factorize with argument that is not not a Series'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        codes, uniques = algos.factorize(data, use_na_sentinel=False)
    tm.assert_numpy_array_equal(uniques, expected_uniques, strict_nan=True)
    tm.assert_numpy_array_equal(codes, expected_codes, strict_nan=True)