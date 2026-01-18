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
def test_multiindex_columns_with_dtypes(self):
    columns = pd.MultiIndex.from_arrays([['one', 'two'], pd.DatetimeIndex(['2017-08-01', '2017-08-02'])], names=['level_1', 'level_2'])
    df = pd.DataFrame([(1, 'a'), (2, 'b'), (3, 'c')], columns=columns)
    _check_pandas_roundtrip(df, preserve_index=True)