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
@pytest.mark.parametrize('keys, expected', [(['b', 'a'], [['b', 'b', 'a', 'a'], [1, 2, 1, 2]]), (['a', 'b'], [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]), ((['a', 'b'], [1, 2]), [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]), ((['a', 'b'], [2, 1]), [['a', 'a', 'b', 'b'], [2, 1, 2, 1]]), ((['b', 'a'], [2, 1]), [['b', 'b', 'a', 'a'], [2, 1, 2, 1]]), ((['b', 'a'], [1, 2]), [['b', 'b', 'a', 'a'], [1, 2, 1, 2]]), ((['c', 'a'], [2, 1]), [['c', 'a', 'a'], [1, 2, 1]])])
@pytest.mark.parametrize('dim', ['index', 'columns'])
def test_loc_getitem_multilevel_index_order(self, dim, keys, expected):
    kwargs = {dim: [['c', 'a', 'a', 'b', 'b'], [1, 1, 2, 1, 2]]}
    df = DataFrame(np.arange(25).reshape(5, 5), **kwargs)
    exp_index = MultiIndex.from_arrays(expected)
    if dim == 'index':
        res = df.loc[keys, :]
        tm.assert_index_equal(res.index, exp_index)
    elif dim == 'columns':
        res = df.loc[:, keys]
        tm.assert_index_equal(res.columns, exp_index)