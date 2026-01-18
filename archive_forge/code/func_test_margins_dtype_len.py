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
def test_margins_dtype_len(self, data):
    mi_val = list(product(['bar', 'foo'], ['one', 'two'])) + [('All', '')]
    mi = MultiIndex.from_tuples(mi_val, names=('A', 'B'))
    expected = DataFrame({'dull': [1, 1, 2, 1, 5], 'shiny': [2, 0, 2, 2, 6]}, index=mi).rename_axis('C', axis=1)
    expected['All'] = expected['dull'] + expected['shiny']
    result = data.pivot_table(values='D', index=['A', 'B'], columns='C', margins=True, aggfunc=len, fill_value=0)
    tm.assert_frame_equal(expected, result)