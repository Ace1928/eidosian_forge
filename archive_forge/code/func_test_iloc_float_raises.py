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
def test_iloc_float_raises(self, series_with_simple_index, frame_or_series, warn_copy_on_write):
    obj = series_with_simple_index
    if frame_or_series is DataFrame:
        obj = obj.to_frame()
    msg = 'Cannot index by location index with a non-integer key'
    with pytest.raises(TypeError, match=msg):
        obj.iloc[3.0]
    with pytest.raises(IndexError, match=_slice_iloc_msg):
        with tm.assert_cow_warning(warn_copy_on_write and frame_or_series is DataFrame):
            obj.iloc[3.0] = 0