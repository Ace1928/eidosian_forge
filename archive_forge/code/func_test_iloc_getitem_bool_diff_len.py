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
@pytest.mark.parametrize('index', [[True, False], [True, False, True, False]])
def test_iloc_getitem_bool_diff_len(self, index):
    s = Series([1, 2, 3])
    msg = f'Boolean index has wrong length: {len(index)} instead of {len(s)}'
    with pytest.raises(IndexError, match=msg):
        s.iloc[index]