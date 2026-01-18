import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_group_diff_object_raises(object_dtype):
    df = DataFrame({'a': ['foo', 'bar', 'bar'], 'b': ['baz', 'foo', 'foo']}, dtype=object_dtype)
    with pytest.raises(TypeError, match='unsupported operand type\\(s\\) for -'):
        df.groupby('a')['b'].diff()