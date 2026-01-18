from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_complex_numbers(using_infer_string):
    df = DataFrame([{'a': 1, 'b': 1 + 1j}, {'a': 1, 'b': 1 + 2j}, {'a': 4, 'b': 1}])
    dtype = 'string[pyarrow_numpy]' if using_infer_string else object
    expected = DataFrame(np.array([1, 1, 1], dtype=np.int64), index=Index([1 + 1j, 1 + 2j, 1 + 0j], name='b'), columns=Index(['a'], dtype=dtype))
    result = df.groupby('b', sort=False).count()
    tm.assert_frame_equal(result, expected)
    expected.index = Index([1 + 0j, 1 + 1j, 1 + 2j], name='b')
    result = df.groupby('b', sort=True).count()
    tm.assert_frame_equal(result, expected)