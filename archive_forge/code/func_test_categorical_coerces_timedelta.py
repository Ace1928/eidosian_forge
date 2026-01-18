from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_coerces_timedelta(all_parsers):
    parser = all_parsers
    dtype = {'b': CategoricalDtype(pd.to_timedelta(['1h', '2h', '3h']))}
    data = 'b\n1h\n2h\n3h'
    expected = DataFrame({'b': Categorical(dtype['b'].categories)})
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)