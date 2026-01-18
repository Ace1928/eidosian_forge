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
@pytest.mark.skipif(Version('1.16.0') <= Version(np.__version__) < Version('1.16.1'), reason='Until numpy/numpy#12745 is resolved')
def test_fixed_offset_timezone(self):
    df = pd.DataFrame({'a': [pd.Timestamp('2012-11-11 00:00:00+01:00'), pd.NaT]})
    _check_pandas_roundtrip(df, check_dtype=False)