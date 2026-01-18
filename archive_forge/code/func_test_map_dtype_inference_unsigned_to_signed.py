import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_dtype_inference_unsigned_to_signed():
    idx = Index([1, 2, 3], dtype=np.uint64)
    result = idx.map(lambda x: -x)
    expected = Index([-1, -2, -3], dtype=np.int64)
    tm.assert_index_equal(result, expected)