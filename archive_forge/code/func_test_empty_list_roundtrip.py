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
def test_empty_list_roundtrip(self):
    empty_list_array = np.empty((3,), dtype=object)
    empty_list_array.fill([])
    df = pd.DataFrame({'a': np.array(['1', '2', '3']), 'b': empty_list_array})
    tbl = pa.Table.from_pandas(df)
    result = tbl.to_pandas()
    tm.assert_frame_equal(result, df)