import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def test_factorize(self, data_for_grouping):
    labels, uniques = pd.factorize(data_for_grouping, use_na_sentinel=True)
    expected_labels = np.array([0, 0, -1, -1, 1, 1, 0], dtype=np.intp)
    expected_uniques = data_for_grouping.take([0, 4])
    tm.assert_numpy_array_equal(labels, expected_labels)
    self.assert_extension_array_equal(uniques, expected_uniques)