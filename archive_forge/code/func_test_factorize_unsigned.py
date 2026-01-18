import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_factorize_unsigned():
    arr = np.array([1, 2, 3], dtype=np.uint64)
    obj = NumpyExtensionArray(arr)
    res_codes, res_unique = obj.factorize()
    exp_codes, exp_unique = pd.factorize(arr)
    tm.assert_numpy_array_equal(res_codes, exp_codes)
    tm.assert_extension_array_equal(res_unique, NumpyExtensionArray(exp_unique))