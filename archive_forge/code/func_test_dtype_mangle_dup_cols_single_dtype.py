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
def test_dtype_mangle_dup_cols_single_dtype(all_parsers):
    parser = all_parsers
    data = 'a,a\n1,1'
    result = parser.read_csv(StringIO(data), dtype=str)
    expected = DataFrame({'a': ['1'], 'a.1': ['1']})
    tm.assert_frame_equal(result, expected)