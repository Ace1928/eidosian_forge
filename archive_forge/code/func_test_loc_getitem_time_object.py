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
def test_loc_getitem_time_object(self, frame_or_series):
    rng = date_range('1/1/2000', '1/5/2000', freq='5min')
    mask = (rng.hour == 9) & (rng.minute == 30)
    obj = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng)
    obj = tm.get_obj(obj, frame_or_series)
    result = obj.loc[time(9, 30)]
    exp = obj.loc[mask]
    tm.assert_equal(result, exp)
    chunk = obj.loc['1/4/2000':]
    result = chunk.loc[time(9, 30)]
    expected = result[-1:]
    result.index = result.index._with_freq(None)
    expected.index = expected.index._with_freq(None)
    tm.assert_equal(result, expected)