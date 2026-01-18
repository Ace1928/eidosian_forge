import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_downcast_dict(self):
    df = DataFrame({'col1': [1, np.nan]})
    msg = "The 'downcast' keyword in fillna"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.fillna({'col1': 2}, downcast={'col1': 'int64'})
    expected = DataFrame({'col1': [1, 2]})
    tm.assert_frame_equal(result, expected)