import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
@pytest.fixture
def np_dtype_to_arrays(any_real_numpy_dtype):
    """
    Fixture returning actual and expected dtype, pandas and numpy arrays and
    mask from a given numpy dtype
    """
    np_dtype = np.dtype(any_real_numpy_dtype)
    pa_type = pa.from_numpy_dtype(np_dtype)
    pa_array = pa.array([0, 1, 2, None], type=pa_type)
    np_expected = np.array([0, 1, 2], dtype=np_dtype)
    mask_expected = np.array([True, True, True, False])
    return (np_dtype, pa_array, np_expected, mask_expected)