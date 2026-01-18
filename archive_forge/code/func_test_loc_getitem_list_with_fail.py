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
def test_loc_getitem_list_with_fail(self):
    s = Series([1, 2, 3])
    s.loc[[2]]
    msg = f""""None of [Index([3], dtype='{np.dtype(int)}')] are in the [index]"""
    with pytest.raises(KeyError, match=re.escape(msg)):
        s.loc[[3]]
    with pytest.raises(KeyError, match='not in index'):
        s.loc[[2, 3]]