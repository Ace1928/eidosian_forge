import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
@pytest.mark.parametrize('data, result_data', [[[tuple('1'), tuple('2')], [10345501319357378243, 8331063931016360761]], [[(1,), (2,)], [9408946347443669104, 3278256261030523334]]])
def test_hash_with_tuple(data, result_data):
    df = DataFrame({'data': data})
    result = hash_pandas_object(df)
    expected = Series(result_data, dtype=np.uint64)
    tm.assert_series_equal(result, expected)