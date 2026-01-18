import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_duplicate_int_indexing(self, indexer_sl):
    ser = Series(range(3), index=[1, 1, 3])
    expected = Series(range(2), index=[1, 1])
    result = indexer_sl(ser)[[1]]
    tm.assert_series_equal(result, expected)