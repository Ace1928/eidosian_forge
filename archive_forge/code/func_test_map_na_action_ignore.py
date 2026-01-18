import warnings
import numpy as np
import pytest
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import (
from pandas.core.arrays.integer import (
from pandas.tests.extension import base
def test_map_na_action_ignore(self, data_missing_for_sorting):
    zero = data_missing_for_sorting[2]
    result = data_missing_for_sorting.map(lambda x: zero, na_action='ignore')
    if data_missing_for_sorting.dtype.kind == 'b':
        expected = np.array([False, pd.NA, False], dtype=object)
    else:
        expected = np.array([zero, np.nan, zero])
    tm.assert_numpy_array_equal(result, expected)