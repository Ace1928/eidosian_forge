from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_dtype_and_names_error(c_parser_only):
    parser = c_parser_only
    data = '\n1.0 1\n2.0 2\n3.0 3\n'
    result = parser.read_csv(StringIO(data), sep='\\s+', header=None)
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]])
    tm.assert_frame_equal(result, expected)
    result = parser.read_csv(StringIO(data), sep='\\s+', header=None, names=['a', 'b'])
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)
    result = parser.read_csv(StringIO(data), sep='\\s+', header=None, names=['a', 'b'], dtype={'a': np.int32})
    expected = DataFrame([[1, 1], [2, 2], [3, 3]], columns=['a', 'b'])
    expected['a'] = expected['a'].astype(np.int32)
    tm.assert_frame_equal(result, expected)
    data = '\n1.0 1\nnan 2\n3.0 3\n'
    warning = RuntimeWarning if np_version_gte1p24 else None
    with pytest.raises(ValueError, match='cannot safely convert'):
        with tm.assert_produces_warning(warning, check_stacklevel=False):
            parser.read_csv(StringIO(data), sep='\\s+', header=None, names=['a', 'b'], dtype={'a': np.int32})