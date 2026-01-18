from collections import OrderedDict
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_from_dict_order_with_single_column(self):
    data = {'alpha': {'value2': 123, 'value1': 532, 'animal': 222, 'plant': False, 'name': 'test'}}
    result = DataFrame.from_dict(data, orient='columns')
    expected = DataFrame([[123], [532], [222], [False], ['test']], index=['value2', 'value1', 'animal', 'plant', 'name'], columns=['alpha'])
    tm.assert_frame_equal(result, expected)