from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_multiindex_with_empty_rangeindex():
    mi = MultiIndex.from_tuples([('B', 1), ('C', 1)])
    df1 = DataFrame([[1, 2]], columns=mi)
    df2 = DataFrame(index=[1], columns=pd.RangeIndex(0))
    result = concat([df1, df2])
    expected = DataFrame([[1, 2], [np.nan, np.nan]], columns=mi)
    tm.assert_frame_equal(result, expected)