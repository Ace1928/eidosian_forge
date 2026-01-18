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
def test_groupby_agg_coercing_bools():
    dat = DataFrame({'a': [1, 1, 2, 2], 'b': [0, 1, 2, 3], 'c': [None, None, 1, 1]})
    gp = dat.groupby('a')
    index = Index([1, 2], name='a')
    result = gp['b'].aggregate(lambda x: (x != 0).all())
    expected = Series([False, True], index=index, name='b')
    tm.assert_series_equal(result, expected)
    result = gp['c'].aggregate(lambda x: x.isnull().all())
    expected = Series([True, False], index=index, name='c')
    tm.assert_series_equal(result, expected)