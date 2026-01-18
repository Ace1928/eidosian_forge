import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs import NaT
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
import pandas as pd
@pytest.mark.parametrize('dtype, fill_value, expected_dtype', [('float32', 1, 'float32'), ('float32', float(np.finfo('float32').max) * 1.1, 'float64'), ('float64', 1, 'float64'), ('float64', float(np.finfo('float32').max) * 1.1, 'float64'), ('complex64', 1, 'complex64'), ('complex64', float(np.finfo('float32').max) * 1.1, 'complex128'), ('complex128', 1, 'complex128'), ('complex128', float(np.finfo('float32').max) * 1.1, 'complex128'), ('float32', 1 + 1j, 'complex64'), ('float32', float(np.finfo('float32').max) * (1.1 + 1j), 'complex128'), ('float64', 1 + 1j, 'complex128'), ('float64', float(np.finfo('float32').max) * (1.1 + 1j), 'complex128'), ('complex64', 1 + 1j, 'complex64'), ('complex64', float(np.finfo('float32').max) * (1.1 + 1j), 'complex128'), ('complex128', 1 + 1j, 'complex128'), ('complex128', float(np.finfo('float32').max) * (1.1 + 1j), 'complex128')])
def test_maybe_promote_float_with_float(dtype, fill_value, expected_dtype):
    dtype = np.dtype(dtype)
    expected_dtype = np.dtype(expected_dtype)
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)