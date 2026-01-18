from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_categorical_dtype_missing(all_parsers):
    parser = all_parsers
    data = 'a,b,c\n1,b,3.4\n1,nan,3.4\n2,a,4.5'
    expected = DataFrame({'a': Categorical(['1', '1', '2']), 'b': Categorical(['b', np.nan, 'a']), 'c': Categorical(['3.4', '3.4', '4.5'])})
    actual = parser.read_csv(StringIO(data), dtype='category')
    tm.assert_frame_equal(actual, expected)