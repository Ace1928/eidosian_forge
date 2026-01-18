import numpy as np
import pytest
from pandas import MultiIndex
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
@pytest.mark.parametrize('null_obj', [np.nan, None])
def test_fillna_null(null_obj, index_or_series_obj):
    obj = index_or_series_obj
    klass = type(obj)
    if not allow_na_ops(obj):
        pytest.skip(f"{klass} doesn't allow for NA operations")
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(obj, MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")
    values = obj._values
    fill_value = values[0]
    expected = values.copy()
    values[0:2] = null_obj
    expected[0:2] = fill_value
    expected = klass(expected)
    obj = klass(values)
    result = obj.fillna(fill_value)
    tm.assert_equal(result, expected)
    assert obj is not result