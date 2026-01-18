from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
def test_from_records_columns_not_modified(self):
    tuples = [(1, 2, 3), (1, 2, 3), (2, 5, 3)]
    columns = ['a', 'b', 'c']
    original_columns = list(columns)
    DataFrame.from_records(tuples, columns=columns, index='a')
    assert columns == original_columns