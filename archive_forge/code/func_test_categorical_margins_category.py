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
def test_categorical_margins_category(self, observed):
    df = DataFrame({'x': np.arange(8), 'y': np.arange(8) // 4, 'z': np.arange(8) % 2})
    expected = DataFrame([[1.0, 2.0, 1.5], [5, 6, 5.5], [3, 4, 3.5]])
    expected.index = Index([0, 1, 'All'], name='y')
    expected.columns = Index([0, 1, 'All'], name='z')
    df.y = df.y.astype('category')
    df.z = df.z.astype('category')
    msg = 'The default value of observed=False is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        table = df.pivot_table('x', 'y', 'z', dropna=observed, margins=True)
    tm.assert_frame_equal(table, expected)