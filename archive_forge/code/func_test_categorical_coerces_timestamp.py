from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_coerces_timestamp(all_parsers):
    parser = all_parsers
    dtype = {'b': CategoricalDtype([Timestamp('2014')])}
    data = 'b\n2014-01-01\n2014-01-01'
    expected = DataFrame({'b': Categorical([Timestamp('2014')] * 2)})
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)