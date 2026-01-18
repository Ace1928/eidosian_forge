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
def test_maybe_promote_any_numpy_dtype_with_na(any_numpy_dtype, nulls_fixture):
    fill_value = nulls_fixture
    dtype = np.dtype(any_numpy_dtype)
    if isinstance(fill_value, Decimal):
        if dtype.kind in 'iufc':
            if dtype.kind in 'iu':
                expected_dtype = np.dtype(np.float64)
            else:
                expected_dtype = dtype
            exp_val_for_scalar = np.nan
        else:
            expected_dtype = np.dtype(object)
            exp_val_for_scalar = fill_value
    elif dtype.kind in 'iu' and fill_value is not NaT:
        expected_dtype = np.float64
        exp_val_for_scalar = np.nan
    elif dtype == object and fill_value is NaT:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    elif dtype.kind in 'mM':
        expected_dtype = dtype
        exp_val_for_scalar = dtype.type('NaT', 'ns')
    elif fill_value is NaT:
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = NaT
    elif dtype.kind in 'fc':
        expected_dtype = dtype
        exp_val_for_scalar = np.nan
    else:
        expected_dtype = np.dtype(object)
        if fill_value is pd.NA:
            exp_val_for_scalar = pd.NA
        else:
            exp_val_for_scalar = np.nan
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)