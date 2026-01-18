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
@pytest.mark.parametrize('start', ['2018', '2020'])
def test_loc_getitem_slice_unordered_dt_index(self, frame_or_series, start):
    obj = frame_or_series([1, 2, 3], index=[Timestamp('2016'), Timestamp('2019'), Timestamp('2017')])
    with pytest.raises(KeyError, match='Value based partial slicing on non-monotonic'):
        obj.loc[start:'2022']