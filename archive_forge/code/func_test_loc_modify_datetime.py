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
def test_loc_modify_datetime(self):
    df = DataFrame.from_dict({'date': [1485264372711, 1485265925110, 1540215845888, 1540282121025]})
    df['date_dt'] = to_datetime(df['date'], unit='ms', cache=True)
    df.loc[:, 'date_dt_cp'] = df.loc[:, 'date_dt']
    df.loc[[2, 3], 'date_dt_cp'] = df.loc[[2, 3], 'date_dt']
    expected = DataFrame([[1485264372711, '2017-01-24 13:26:12.711', '2017-01-24 13:26:12.711'], [1485265925110, '2017-01-24 13:52:05.110', '2017-01-24 13:52:05.110'], [1540215845888, '2018-10-22 13:44:05.888', '2018-10-22 13:44:05.888'], [1540282121025, '2018-10-23 08:08:41.025', '2018-10-23 08:08:41.025']], columns=['date', 'date_dt', 'date_dt_cp'])
    columns = ['date_dt', 'date_dt_cp']
    expected[columns] = expected[columns].apply(to_datetime)
    tm.assert_frame_equal(df, expected)