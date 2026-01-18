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
@td.skip_array_manager_not_yet_implemented
def test_iloc_getitem_int_single_ea_block_view(self):
    arr = interval_range(1, 10.0)._values
    df = DataFrame(arr)
    ser = df.iloc[2]
    assert arr[2] != arr[-1]
    arr[2] = arr[-1]
    assert ser[0] == arr[-1]