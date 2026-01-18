import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_add_2d(dtype, request, arrow_string_storage):
    if dtype.storage in arrow_string_storage:
        reason = "Failed: DID NOT RAISE <class 'ValueError'>"
        mark = pytest.mark.xfail(raises=None, reason=reason)
        request.applymarker(mark)
    a = pd.array(['a', 'b', 'c'], dtype=dtype)
    b = np.array([['a', 'b', 'c']], dtype=object)
    with pytest.raises(ValueError, match='3 != 1'):
        a + b
    s = pd.Series(a)
    with pytest.raises(ValueError, match='3 != 1'):
        s + b