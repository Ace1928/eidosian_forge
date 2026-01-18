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
def test_loc_copy_vs_view(self, request, using_copy_on_write):
    if not using_copy_on_write:
        mark = pytest.mark.xfail(reason='accidental fix reverted - GH37497')
        request.applymarker(mark)
    x = DataFrame(zip(range(3), range(3)), columns=['a', 'b'])
    y = x.copy()
    q = y.loc[:, 'a']
    q += 2
    tm.assert_frame_equal(x, y)
    z = x.copy()
    q = z.loc[x.index, 'a']
    q += 2
    tm.assert_frame_equal(x, z)