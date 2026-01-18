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
def test_loc_getitem_series_label_list_missing_values(self):
    key = np.array(['2001-01-04', '2001-01-02', '2001-01-04', '2001-01-14'], dtype='datetime64')
    ser = Series([2, 5, 8, 11], date_range('2001-01-01', freq='D', periods=4))
    with pytest.raises(KeyError, match='not in index'):
        ser.loc[key]