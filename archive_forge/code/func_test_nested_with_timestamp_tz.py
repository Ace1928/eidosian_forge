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
def test_nested_with_timestamp_tz():
    ts = pd.Timestamp.now()
    ts_dt = ts.to_pydatetime()
    for unit in ['s', 'ms', 'us']:
        if unit in ['s', 'ms']:

            def truncate(x):
                return x.replace(microsecond=0)
        else:

            def truncate(x):
                return x
        arr = pa.array([ts], type=pa.timestamp(unit))
        arr2 = pa.array([ts], type=pa.timestamp(unit, tz='America/New_York'))
        arr3 = pa.StructArray.from_arrays([arr, arr], ['start', 'stop'])
        arr4 = pa.StructArray.from_arrays([arr2, arr2], ['start', 'stop'])
        result = arr3.to_pandas()
        assert isinstance(result[0]['start'], datetime)
        assert result[0]['start'].tzinfo is None
        assert isinstance(result[0]['stop'], datetime)
        assert result[0]['stop'].tzinfo is None
        result = arr4.to_pandas()
        assert isinstance(result[0]['start'], datetime)
        assert result[0]['start'].tzinfo is not None
        utc_dt = result[0]['start'].astimezone(timezone.utc)
        assert truncate(utc_dt).replace(tzinfo=None) == truncate(ts_dt)
        assert isinstance(result[0]['stop'], datetime)
        assert result[0]['stop'].tzinfo is not None
        result = pa.table({'a': arr3}).to_pandas()
        assert isinstance(result['a'][0]['start'], datetime)
        assert result['a'][0]['start'].tzinfo is None
        assert isinstance(result['a'][0]['stop'], datetime)
        assert result['a'][0]['stop'].tzinfo is None
        result = pa.table({'a': arr4}).to_pandas()
        assert isinstance(result['a'][0]['start'], datetime)
        assert result['a'][0]['start'].tzinfo is not None
        assert isinstance(result['a'][0]['stop'], datetime)
        assert result['a'][0]['stop'].tzinfo is not None