from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_trivial(using_infer_string):
    df = DataFrame({'key': ['a', 'a', 'b', 'b', 'a'], 'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['key', 'data'])
    dtype = 'string' if using_infer_string else 'object'
    expected = pd.concat([df.iloc[1:], df.iloc[1:]], axis=1, keys=['float64', dtype])
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby([str(x) for x in df.dtypes], axis=1)
    result = gb.apply(lambda x: df.iloc[1:])
    tm.assert_frame_equal(result, expected)