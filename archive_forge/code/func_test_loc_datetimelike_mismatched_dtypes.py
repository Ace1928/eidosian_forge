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
def test_loc_datetimelike_mismatched_dtypes():
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['a', 'b', 'c'], index=date_range('2012', freq='h', periods=5))
    df = df.iloc[[0, 2, 2, 3]].copy()
    dti = df.index
    tdi = pd.TimedeltaIndex(dti.asi8)
    msg = 'None of \\[TimedeltaIndex.* are in the \\[index\\]'
    with pytest.raises(KeyError, match=msg):
        df.loc[tdi]
    with pytest.raises(KeyError, match=msg):
        df['a'].loc[tdi]