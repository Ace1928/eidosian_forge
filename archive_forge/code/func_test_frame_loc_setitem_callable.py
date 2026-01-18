from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_frame_loc_setitem_callable(self):
    df = DataFrame({'X': [1, 2, 3, 4], 'Y': Series(list('aabb'), dtype=object)}, index=list('ABCD'))
    res = df.copy()
    res.loc[lambda x: ['A', 'C']] = -20
    exp = df.copy()
    exp.loc[['A', 'C']] = -20
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.loc[lambda x: ['A', 'C'], :] = 20
    exp = df.copy()
    exp.loc[['A', 'C'], :] = 20
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.loc[lambda x: ['A', 'C'], lambda x: 'X'] = -1
    exp = df.copy()
    exp.loc[['A', 'C'], 'X'] = -1
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.loc[lambda x: ['A', 'C'], lambda x: ['X']] = [5, 10]
    exp = df.copy()
    exp.loc[['A', 'C'], ['X']] = [5, 10]
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.loc[['A', 'C'], lambda x: 'X'] = np.array([-1, -2])
    exp = df.copy()
    exp.loc[['A', 'C'], 'X'] = np.array([-1, -2])
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.loc[['A', 'C'], lambda x: ['X']] = 10
    exp = df.copy()
    exp.loc[['A', 'C'], ['X']] = 10
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.loc[lambda x: ['A', 'C'], 'X'] = -2
    exp = df.copy()
    exp.loc[['A', 'C'], 'X'] = -2
    tm.assert_frame_equal(res, exp)
    res = df.copy()
    res.loc[lambda x: ['A', 'C'], ['X']] = -4
    exp = df.copy()
    exp.loc[['A', 'C'], ['X']] = -4
    tm.assert_frame_equal(res, exp)