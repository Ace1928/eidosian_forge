from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_frame_iloc_setitem_callable(self):
    df = DataFrame({'X': [1, 2, 3, 4], 'Y': Series(list('aabb'), dtype=object)}, index=list('ABCD'))
    res = df.copy()
    res.iloc[lambda x: [1, 3]] = 0
    exp = df.copy()
    exp.iloc[[1, 3]] = 0
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.iloc[lambda x: [1, 3], :] = -1
    exp = df.copy()
    exp.iloc[[1, 3], :] = -1
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.iloc[lambda x: [1, 3], lambda x: 0] = 5
    exp = df.copy()
    exp.iloc[[1, 3], 0] = 5
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.iloc[lambda x: [1, 3], lambda x: [0]] = 25
    exp = df.copy()
    exp.iloc[[1, 3], [0]] = 25
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.iloc[[1, 3], lambda x: 0] = -3
    exp = df.copy()
    exp.iloc[[1, 3], 0] = -3
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.iloc[[1, 3], lambda x: [0]] = -5
    exp = df.copy()
    exp.iloc[[1, 3], [0]] = -5
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.iloc[lambda x: [1, 3], 0] = 10
    exp = df.copy()
    exp.iloc[[1, 3], 0] = 10
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.iloc[lambda x: [1, 3], [0]] = [-5, -5]
    exp = df.copy()
    exp.iloc[[1, 3], [0]] = [-5, -5]
    tm.assert_frame_equal(res, exp)