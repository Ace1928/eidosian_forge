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
def test_loc_setitem_consistency_dt64_to_str(self, frame_for_consistency):
    expected = DataFrame({'date': Series('foo', index=range(5)), 'val': Series(range(5), dtype=np.int64)})
    df = frame_for_consistency.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.loc[:, 'date'] = 'foo'
    tm.assert_frame_equal(df, expected)