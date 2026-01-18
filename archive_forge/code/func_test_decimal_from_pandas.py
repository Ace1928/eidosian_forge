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
@pytest.mark.parametrize(('values', 'expected_type'), [pytest.param(decimal32, pa.decimal128(7, 3), id='decimal32'), pytest.param(decimal64, pa.decimal128(12, 6), id='decimal64'), pytest.param(decimal128, pa.decimal128(26, 11), id='decimal128')])
def test_decimal_from_pandas(self, values, expected_type):
    expected = pd.DataFrame({'decimals': values})
    table = pa.Table.from_pandas(expected, preserve_index=False)
    field = pa.field('decimals', expected_type)
    expected_schema = pa.schema([field], metadata=table.schema.metadata)
    assert table.schema.equals(expected_schema)