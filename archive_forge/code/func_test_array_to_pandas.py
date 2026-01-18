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
def test_array_to_pandas():
    if Version(pd.__version__) < Version('1.1'):
        pytest.skip('ExtensionDtype to_pandas method missing')
    for arr in [pd.period_range('2012-01-01', periods=3, freq='D').array, pd.interval_range(1, 4).array]:
        result = pa.array(arr).to_pandas()
        expected = pd.Series(arr)
        tm.assert_series_equal(result, expected)
        result = pa.table({'col': arr})['col'].to_pandas()
        expected = pd.Series(arr, name='col')
        tm.assert_series_equal(result, expected)