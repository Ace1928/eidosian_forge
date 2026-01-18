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
def test_from_records_sequencelike_empty(self):
    result = DataFrame.from_records([], columns=['foo', 'bar', 'baz'])
    assert len(result) == 0
    tm.assert_index_equal(result.columns, Index(['foo', 'bar', 'baz']))
    result = DataFrame.from_records([])
    assert len(result) == 0
    assert len(result.columns) == 0