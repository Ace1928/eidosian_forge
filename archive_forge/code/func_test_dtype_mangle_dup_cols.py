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
@pytest.mark.parametrize('dtypes, exp_value', [({}, '1'), ({'a.1': 'int64'}, 1)])
def test_dtype_mangle_dup_cols(all_parsers, dtypes, exp_value):
    parser = all_parsers
    data = 'a,a\n1,1'
    dtype_dict = {'a': str, **dtypes}
    dtype_dict_copy = dtype_dict.copy()
    result = parser.read_csv(StringIO(data), dtype=dtype_dict)
    expected = DataFrame({'a': ['1'], 'a.1': [exp_value]})
    assert dtype_dict == dtype_dict_copy, 'dtype dict changed'
    tm.assert_frame_equal(result, expected)