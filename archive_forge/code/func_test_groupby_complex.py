import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('func, output', [('mean', [8 + 18j, 10 + 22j]), ('sum', [40 + 90j, 50 + 110j])])
def test_groupby_complex(func, output):
    data = Series(np.arange(20).reshape(10, 2).dot([1, 2j]))
    result = data.groupby(data.index % 2).agg(func)
    expected = Series(output)
    tm.assert_series_equal(result, expected)