from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('arr, correct', [('arr_float', False), ('arr_nan', False), ('arr_float_nan', False), ('arr_nan_nan', False), ('arr_float_inf', True), ('arr_inf', True), ('arr_nan_inf', True), ('arr_float_nan_inf', True), ('arr_nan_nan_inf', True)])
@pytest.mark.parametrize('astype', [None, 'f4', 'f2'])
def test_has_infs_floats(request, arr, correct, astype, disable_bottleneck):
    val = request.getfixturevalue(arr)
    if astype is not None:
        val = val.astype(astype)
    while getattr(val, 'ndim', True):
        res0 = nanops._has_infs(val)
        if correct:
            assert res0
        else:
            assert not res0
        if not hasattr(val, 'ndim'):
            break
        val = np.take(val, 0, axis=-1)