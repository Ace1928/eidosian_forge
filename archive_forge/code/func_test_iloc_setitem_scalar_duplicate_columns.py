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
def test_iloc_setitem_scalar_duplicate_columns(self):
    df1 = DataFrame([{'A': None, 'B': 1}, {'A': 2, 'B': 2}])
    df2 = DataFrame([{'A': 3, 'B': 3}, {'A': 4, 'B': 4}])
    df = concat([df1, df2], axis=1)
    df.iloc[0, 0] = -1
    assert df.iloc[0, 0] == -1
    assert df.iloc[0, 2] == 3
    assert df.dtypes.iloc[2] == np.int64