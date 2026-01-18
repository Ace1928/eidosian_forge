from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('right_vals', [['foo', 'bar'], Series(['foo', 'bar']).astype('category')])
def test_different(self, right_vals):
    left = DataFrame({'A': ['foo', 'bar'], 'B': Series(['foo', 'bar']).astype('category'), 'C': [1, 2], 'D': [1.0, 2.0], 'E': Series([1, 2], dtype='uint64'), 'F': Series([1, 2], dtype='int32')})
    right = DataFrame({'A': right_vals})
    result = merge(left, right, on='A')
    assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)