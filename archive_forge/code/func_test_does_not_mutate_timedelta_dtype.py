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
def test_does_not_mutate_timedelta_dtype():
    expected = np.dtype('m8')
    assert np.dtype(np.timedelta64) == expected
    df = pd.DataFrame({'a': [np.timedelta64()]})
    t = pa.Table.from_pandas(df)
    t.to_pandas()
    assert np.dtype(np.timedelta64) == expected