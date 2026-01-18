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
def test_loc_setitem_multiindex_timestamp():
    vals = np.random.default_rng(2).standard_normal((8, 6))
    idx = date_range('1/1/2000', periods=8)
    cols = ['A', 'B', 'C', 'D', 'E', 'F']
    exp = DataFrame(vals, index=idx, columns=cols)
    exp.loc[exp.index[1], ('A', 'B')] = np.nan
    vals[1][0:2] = np.nan
    res = DataFrame(vals, index=idx, columns=cols)
    tm.assert_frame_equal(res, exp)