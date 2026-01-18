from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtypes_defaultdict_mangle_dup_cols(all_parsers):
    data = 'a,b,a,b,b.1\n1,2,3,4,5\n'
    dtype = defaultdict(lambda: 'float64', a='int64')
    dtype['b.1'] = 'int64'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), dtype=dtype)
    expected = DataFrame({'a': [1], 'b': [2.0], 'a.1': [3], 'b.2': [4.0], 'b.1': [5]})
    tm.assert_frame_equal(result, expected)