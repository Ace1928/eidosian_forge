from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('insert, coerced_val, coerced_dtype', [(1, 1, None), (1.1, 1.1, np.float64), (False, False, object), ('x', 'x', object)])
def test_insert_int_index(self, any_int_numpy_dtype, insert, coerced_val, coerced_dtype):
    dtype = any_int_numpy_dtype
    obj = pd.Index([1, 2, 3, 4], dtype=dtype)
    coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype
    exp = pd.Index([1, coerced_val, 2, 3, 4], dtype=coerced_dtype)
    self._assert_insert_conversion(obj, insert, exp, coerced_dtype)