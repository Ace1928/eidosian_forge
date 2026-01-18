from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('dtype', [str, object])
@pytest.mark.parametrize('check_orig', [True, False])
@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtype_all_columns(all_parsers, dtype, check_orig):
    parser = all_parsers
    df = DataFrame(np.random.default_rng(2).random((5, 2)).round(4), columns=list('AB'), index=['1A', '1B', '1C', '1D', '1E'])
    with tm.ensure_clean('__passing_str_as_dtype__.csv') as path:
        df.to_csv(path)
        result = parser.read_csv(path, dtype=dtype, index_col=0)
        if check_orig:
            expected = df.copy()
            result = result.astype(float)
        else:
            expected = df.astype(str)
        tm.assert_frame_equal(result, expected)