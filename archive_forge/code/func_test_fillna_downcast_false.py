import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_downcast_false(self, frame_or_series):
    obj = frame_or_series([1, 2, 3], dtype='object')
    msg = "The 'downcast' keyword in fillna"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = obj.fillna('', downcast=False)
    tm.assert_equal(result, obj)