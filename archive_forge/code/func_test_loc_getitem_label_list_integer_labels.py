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
@pytest.mark.parametrize('columns, column_key, expected_columns', [([2011, 2012, 2013], [2011, 2012], [0, 1]), ([2011, 2012, 'All'], [2011, 2012], [0, 1]), ([2011, 2012, 'All'], [2011, 'All'], [0, 2])])
def test_loc_getitem_label_list_integer_labels(columns, column_key, expected_columns):
    df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=columns, index=list('ABC'))
    expected = df.iloc[:, expected_columns]
    result = df.loc[['A', 'B', 'C'], column_key]
    tm.assert_frame_equal(result, expected, check_column_type=True)