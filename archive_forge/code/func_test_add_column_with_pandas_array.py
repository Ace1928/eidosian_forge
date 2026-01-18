from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_add_column_with_pandas_array(self):
    df = DataFrame({'a': [1, 2, 3, 4], 'b': ['a', 'b', 'c', 'd']})
    df['c'] = pd.arrays.NumpyExtensionArray(np.array([1, 2, None, 3], dtype=object))
    df2 = DataFrame({'a': [1, 2, 3, 4], 'b': ['a', 'b', 'c', 'd'], 'c': pd.arrays.NumpyExtensionArray(np.array([1, 2, None, 3], dtype=object))})
    assert type(df['c']._mgr.blocks[0]) == NumpyBlock
    assert df['c']._mgr.blocks[0].is_object
    assert type(df2['c']._mgr.blocks[0]) == NumpyBlock
    assert df2['c']._mgr.blocks[0].is_object
    tm.assert_frame_equal(df, df2)