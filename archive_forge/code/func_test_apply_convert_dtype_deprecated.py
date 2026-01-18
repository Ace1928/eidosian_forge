import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('convert_dtype', [True, False])
def test_apply_convert_dtype_deprecated(convert_dtype):
    ser = Series(np.random.default_rng(2).standard_normal(10))

    def func(x):
        return x if x > 0 else np.nan
    with tm.assert_produces_warning(FutureWarning):
        ser.apply(func, convert_dtype=convert_dtype, by_row='compat')