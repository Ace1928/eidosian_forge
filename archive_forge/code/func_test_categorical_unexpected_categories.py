from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_unexpected_categories(all_parsers):
    parser = all_parsers
    dtype = {'b': CategoricalDtype(['a', 'b', 'd', 'e'])}
    data = 'b\nd\na\nc\nd'
    expected = DataFrame({'b': Categorical(list('dacd'), dtype=dtype['b'])})
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)