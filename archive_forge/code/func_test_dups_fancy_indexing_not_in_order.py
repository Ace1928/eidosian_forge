import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_dups_fancy_indexing_not_in_order(self):
    df = DataFrame({'test': [5, 7, 9, 11], 'test1': [4.0, 5, 6, 7], 'other': list('abcd')}, index=['A', 'A', 'B', 'C'])
    rows = ['C', 'B']
    expected = DataFrame({'test': [11, 9], 'test1': [7.0, 6], 'other': ['d', 'c']}, index=rows)
    result = df.loc[rows]
    tm.assert_frame_equal(result, expected)
    result = df.loc[Index(rows)]
    tm.assert_frame_equal(result, expected)
    rows = ['C', 'B', 'E']
    with pytest.raises(KeyError, match='not in index'):
        df.loc[rows]
    rows = ['F', 'G', 'H', 'C', 'B', 'E']
    with pytest.raises(KeyError, match='not in index'):
        df.loc[rows]