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
def test_dtype_multi_index(all_parsers):
    parser = all_parsers
    data = 'A,B,B\nX,Y,Z\n1,2,3'
    result = parser.read_csv(StringIO(data), header=list(range(2)), dtype={('A', 'X'): np.int32, ('B', 'Y'): np.int32, ('B', 'Z'): np.float32})
    expected = DataFrame({('A', 'X'): np.int32([1]), ('B', 'Y'): np.int32([2]), ('B', 'Z'): np.float32([3])})
    tm.assert_frame_equal(result, expected)