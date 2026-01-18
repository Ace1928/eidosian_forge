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
def test_dtype_per_column(all_parsers):
    parser = all_parsers
    data = 'one,two\n1,2.5\n2,3.5\n3,4.5\n4,5.5'
    expected = DataFrame([[1, '2.5'], [2, '3.5'], [3, '4.5'], [4, '5.5']], columns=['one', 'two'])
    expected['one'] = expected['one'].astype(np.float64)
    expected['two'] = expected['two'].astype(object)
    result = parser.read_csv(StringIO(data), dtype={'one': np.float64, 1: str})
    tm.assert_frame_equal(result, expected)