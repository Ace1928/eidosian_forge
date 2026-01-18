from collections import OrderedDict
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_from_dict_columns_parameter(self):
    result = DataFrame.from_dict(OrderedDict([('A', [1, 2]), ('B', [4, 5])]), orient='index', columns=['one', 'two'])
    expected = DataFrame([[1, 2], [4, 5]], index=['A', 'B'], columns=['one', 'two'])
    tm.assert_frame_equal(result, expected)
    msg = "cannot use columns parameter with orient='columns'"
    with pytest.raises(ValueError, match=msg):
        DataFrame.from_dict({'A': [1, 2], 'B': [4, 5]}, orient='columns', columns=['one', 'two'])
    with pytest.raises(ValueError, match=msg):
        DataFrame.from_dict({'A': [1, 2], 'B': [4, 5]}, columns=['one', 'two'])