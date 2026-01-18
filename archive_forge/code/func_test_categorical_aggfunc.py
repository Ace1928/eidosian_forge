from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_categorical_aggfunc(self, observed):
    df = DataFrame({'C1': ['A', 'B', 'C', 'C'], 'C2': ['a', 'a', 'b', 'b'], 'V': [1, 2, 3, 4]})
    df['C1'] = df['C1'].astype('category')
    msg = 'The default value of observed=False is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.pivot_table('V', index='C1', columns='C2', dropna=observed, aggfunc='count')
    expected_index = pd.CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'], ordered=False, name='C1')
    expected_columns = Index(['a', 'b'], name='C2')
    expected_data = np.array([[1, 0], [1, 0], [0, 2]], dtype=np.int64)
    expected = DataFrame(expected_data, index=expected_index, columns=expected_columns)
    tm.assert_frame_equal(result, expected)