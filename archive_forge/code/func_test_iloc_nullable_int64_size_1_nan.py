from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_nullable_int64_size_1_nan(self):
    result = DataFrame({'a': ['test'], 'b': [np.nan]})
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        result.loc[:, 'b'] = result.loc[:, 'b'].astype('Int64')
    expected = DataFrame({'a': ['test'], 'b': array([NA], dtype='Int64')})
    tm.assert_frame_equal(result, expected)