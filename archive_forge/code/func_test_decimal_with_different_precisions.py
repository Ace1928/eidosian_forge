import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_decimal_with_different_precisions(self):
    data = [decimal.Decimal('0.01'), decimal.Decimal('0.001')]
    series = pd.Series(data)
    array = pa.array(series)
    assert array.to_pylist() == data
    assert array.type == pa.decimal128(3, 3)
    array = pa.array(data, type=pa.decimal128(12, 5))
    expected = [decimal.Decimal('0.01000'), decimal.Decimal('0.00100')]
    assert array.to_pylist() == expected