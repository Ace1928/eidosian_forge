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
def test_loc_setitem_mask_and_label_with_datetimeindex(self):
    df = DataFrame(np.arange(6.0).reshape(3, 2), columns=list('AB'), index=date_range('1/1/2000', periods=3, freq='1h'))
    expected = df.copy()
    expected['C'] = [expected.index[0]] + [pd.NaT, pd.NaT]
    mask = df.A < 1
    df.loc[mask, 'C'] = df.loc[mask].index
    tm.assert_frame_equal(df, expected)