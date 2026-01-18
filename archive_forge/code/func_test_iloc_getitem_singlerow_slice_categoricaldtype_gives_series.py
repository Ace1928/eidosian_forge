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
def test_iloc_getitem_singlerow_slice_categoricaldtype_gives_series(self):
    df = DataFrame({'x': Categorical('a b c d e'.split())})
    result = df.iloc[0]
    raw_cat = Categorical(['a'], categories=['a', 'b', 'c', 'd', 'e'])
    expected = Series(raw_cat, index=['x'], name=0, dtype='category')
    tm.assert_series_equal(result, expected)