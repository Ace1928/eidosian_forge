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
def test_column_index_names_with_tz(self):
    df = pd.DataFrame(np.random.randn(5, 3), columns=pd.date_range('2021-01-01', periods=3, freq='50D', tz='CET'))
    _check_pandas_roundtrip(df, preserve_index=True)