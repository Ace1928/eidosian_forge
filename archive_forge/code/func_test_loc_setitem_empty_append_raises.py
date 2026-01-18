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
def test_loc_setitem_empty_append_raises(self):
    data = [1, 2]
    df = DataFrame(columns=['x', 'y'])
    df.index = df.index.astype(np.int64)
    msg = f"None of \\[Index\\(\\[0, 1\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]"
    with pytest.raises(KeyError, match=msg):
        df.loc[[0, 1], 'x'] = data
    msg = 'setting an array element with a sequence.'
    with pytest.raises(ValueError, match=msg):
        df.loc[0:2, 'x'] = data