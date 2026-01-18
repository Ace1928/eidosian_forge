import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def test_decimal_array_with_none_and_nan():
    values = [decimal.Decimal('1.234'), None, np.nan, decimal.Decimal('nan')]
    with pytest.raises(TypeError):
        array = pa.array(values)
    array = pa.array(values, from_pandas=True)
    assert array.type == pa.decimal128(4, 3)
    assert array.to_pylist() == values[:2] + [None, None]
    array = pa.array(values, type=pa.decimal128(10, 4), from_pandas=True)
    assert array.to_pylist() == [decimal.Decimal('1.2340'), None, None, None]