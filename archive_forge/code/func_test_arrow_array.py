import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
def test_arrow_array(data):
    arr = pa.array(data)
    expected = pa.array(data.to_numpy(object, na_value=None), type=pa.from_numpy_dtype(data.dtype.numpy_dtype))
    assert arr.equals(expected)