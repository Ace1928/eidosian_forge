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
@pytest.mark.parametrize('values', [pytest.param(decimal32, id='decimal32'), pytest.param(decimal64, id='decimal64'), pytest.param(decimal128, id='decimal128')])
def test_decimal_to_pandas(self, values):
    expected = pd.DataFrame({'decimals': values})
    converted = pa.Table.from_pandas(expected)
    df = converted.to_pandas()
    tm.assert_frame_equal(df, expected)