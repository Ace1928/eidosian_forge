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
@pytest.mark.parametrize('default', ['float', 'float64'])
def test_dtypes_defaultdict(all_parsers, default):
    data = 'a,b\n1,2\n'
    dtype = defaultdict(lambda: default, a='int64')
    parser = all_parsers
    result = parser.read_csv(StringIO(data), dtype=dtype)
    expected = DataFrame({'a': [1], 'b': 2.0})
    tm.assert_frame_equal(result, expected)