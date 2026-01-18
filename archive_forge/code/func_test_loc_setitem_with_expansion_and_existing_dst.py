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
def test_loc_setitem_with_expansion_and_existing_dst(self):
    start = Timestamp('2017-10-29 00:00:00+0200', tz='Europe/Madrid')
    end = Timestamp('2017-10-29 03:00:00+0100', tz='Europe/Madrid')
    ts = Timestamp('2016-10-10 03:00:00', tz='Europe/Madrid')
    idx = date_range(start, end, inclusive='left', freq='h')
    assert ts not in idx
    result = DataFrame(index=idx, columns=['value'])
    result.loc[ts, 'value'] = 12
    expected = DataFrame([np.nan] * len(idx) + [12], index=idx.append(DatetimeIndex([ts])), columns=['value'], dtype=object)
    tm.assert_frame_equal(result, expected)