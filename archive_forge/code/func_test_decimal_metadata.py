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
def test_decimal_metadata(self):
    expected = pd.DataFrame({'decimals': [decimal.Decimal('394092382910493.12341234678'), -decimal.Decimal('314292388910493.12343437128')]})
    table = pa.Table.from_pandas(expected)
    js = table.schema.pandas_metadata
    assert 'mixed' not in js
    data_column = js['columns'][0]
    assert data_column['pandas_type'] == 'decimal'
    assert data_column['numpy_type'] == 'object'
    assert data_column['metadata'] == {'precision': 26, 'scale': 11}