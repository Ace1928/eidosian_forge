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
def test_getitem_loc_str_periodindex(self):
    msg = 'Period with BDay freq is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        index = pd.period_range(start='2000', periods=20, freq='B')
        series = Series(range(20), index=index)
        assert series.loc['2000-01-14'] == 9