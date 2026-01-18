from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('method', ['index', 'rindex'])
def test_index_wrong_type_raises(index_or_series, any_string_dtype, method):
    obj = index_or_series([], dtype=any_string_dtype)
    msg = 'expected a string object, not int'
    with pytest.raises(TypeError, match=msg):
        getattr(obj.str, method)(0)