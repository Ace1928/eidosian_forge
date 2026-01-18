from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('values,inferred_type', [(['a', 'b'], 'string'), (['a', 'b', 1], 'mixed-integer'), (['a', 'b', 1.3], 'mixed'), (['a', 'b', 1.3, 1], 'mixed-integer'), (['aa', datetime(2011, 1, 1)], 'mixed')])
def test_index_str_accessor_visibility(values, inferred_type, index_or_series):
    obj = index_or_series(values)
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type
    assert isinstance(obj.str, StringMethods)