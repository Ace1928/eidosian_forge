import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('values', ['baz'])
def test_get_dummies_with_string_values(self, values):
    df = DataFrame({'bar': [1, 2, 3, 4, 5, 6], 'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'baz': ['A', 'B', 'C', 'A', 'B', 'C'], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    msg = 'Input must be a list-like for parameter `columns`'
    with pytest.raises(TypeError, match=msg):
        get_dummies(df, columns=values)