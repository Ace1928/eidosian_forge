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
@pytest.mark.parametrize('data, expected_codes, expected_uniques', [(Index(Categorical(['a', 'a', 'b'])), np.array([0, 0, 1], dtype=np.intp), CategoricalIndex(['a', 'b'], categories=['a', 'b'], dtype='category')), (Series(Categorical(['a', 'a', 'b'])), np.array([0, 0, 1], dtype=np.intp), CategoricalIndex(['a', 'b'], categories=['a', 'b'], dtype='category')), (Series(DatetimeIndex(['2017', '2017'], tz='US/Eastern')), np.array([0, 0], dtype=np.intp), DatetimeIndex(['2017'], tz='US/Eastern'))])
def test_factorize_mixed_values(self, data, expected_codes, expected_uniques):
    codes, uniques = algos.factorize(data)
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_index_equal(uniques, expected_uniques)