import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_convert_almost_null_slice(self, index):
    key = slice(None, None, 'foo')
    if isinstance(index, IntervalIndex):
        msg = 'label-based slicing with step!=1 is not supported for IntervalIndex'
        with pytest.raises(ValueError, match=msg):
            index._convert_slice_indexer(key, 'loc')
    else:
        msg = "'>=' not supported between instances of 'str' and 'int'"
        with pytest.raises(TypeError, match=msg):
            index._convert_slice_indexer(key, 'loc')