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
def test_column_types_consistent(self):
    df = DataFrame(data={'channel': [1, 2, 3], 'A': ['String 1', np.nan, 'String 2'], 'B': [Timestamp('2019-06-11 11:00:00'), pd.NaT, Timestamp('2019-06-11 12:00:00')]})
    df2 = DataFrame(data={'A': ['String 3'], 'B': [Timestamp('2019-06-11 12:00:00')]})
    df.loc[df['A'].isna(), ['A', 'B']] = df2.values
    expected = DataFrame(data={'channel': [1, 2, 3], 'A': ['String 1', 'String 3', 'String 2'], 'B': [Timestamp('2019-06-11 11:00:00'), Timestamp('2019-06-11 12:00:00'), Timestamp('2019-06-11 12:00:00')]})
    tm.assert_frame_equal(df, expected)