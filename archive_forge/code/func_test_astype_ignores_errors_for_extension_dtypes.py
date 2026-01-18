import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data, dtype', [(['x', 'y', 'z'], 'string[python]'), pytest.param(['x', 'y', 'z'], 'string[pyarrow]', marks=td.skip_if_no('pyarrow')), (['x', 'y', 'z'], 'category'), (3 * [Timestamp('2020-01-01', tz='UTC')], None), (3 * [Interval(0, 1)], None)])
@pytest.mark.parametrize('errors', ['raise', 'ignore'])
def test_astype_ignores_errors_for_extension_dtypes(self, data, dtype, errors):
    df = DataFrame(Series(data, dtype=dtype))
    if errors == 'ignore':
        expected = df
        result = df.astype(float, errors=errors)
        tm.assert_frame_equal(result, expected)
    else:
        msg = '(Cannot cast)|(could not convert)'
        with pytest.raises((ValueError, TypeError), match=msg):
            df.astype(float, errors=errors)