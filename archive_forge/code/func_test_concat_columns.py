import itertools
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_concat_columns(self, data, na_value):
    df1 = pd.DataFrame({'A': data[:3]})
    df2 = pd.DataFrame({'B': [1, 2, 3]})
    expected = pd.DataFrame({'A': data[:3], 'B': [1, 2, 3]})
    result = pd.concat([df1, df2], axis=1)
    tm.assert_frame_equal(result, expected)
    result = pd.concat([df1['A'], df2['B']], axis=1)
    tm.assert_frame_equal(result, expected)
    df2 = pd.DataFrame({'B': [1, 2, 3]}, index=[1, 2, 3])
    expected = pd.DataFrame({'A': data._from_sequence(list(data[:3]) + [na_value], dtype=data.dtype), 'B': [np.nan, 1, 2, 3]})
    result = pd.concat([df1, df2], axis=1)
    tm.assert_frame_equal(result, expected)
    result = pd.concat([df1['A'], df2['B']], axis=1)
    tm.assert_frame_equal(result, expected)