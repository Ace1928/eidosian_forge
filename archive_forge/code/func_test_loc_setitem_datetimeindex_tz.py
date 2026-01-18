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
@pytest.mark.parametrize('idxer', ['var', ['var']])
def test_loc_setitem_datetimeindex_tz(self, idxer, tz_naive_fixture):
    tz = tz_naive_fixture
    idx = date_range(start='2015-07-12', periods=3, freq='h', tz=tz)
    expected = DataFrame(1.2, index=idx, columns=['var'])
    result = DataFrame(index=idx, columns=['var'], dtype=np.float64)
    with tm.assert_produces_warning(FutureWarning if idxer == 'var' else None, match='incompatible dtype'):
        result.loc[:, idxer] = expected
    tm.assert_frame_equal(result, expected)