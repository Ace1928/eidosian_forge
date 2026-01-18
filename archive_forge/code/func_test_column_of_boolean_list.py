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
def test_column_of_boolean_list(self):
    array = pa.array([[True, False], [True]], type=pa.list_(pa.bool_()))
    table = pa.Table.from_arrays([array], names=['col1'])
    df = table.to_pandas()
    expected_df = pd.DataFrame({'col1': [[True, False], [True]]})
    tm.assert_frame_equal(df, expected_df)
    s = table[0].to_pandas()
    tm.assert_series_equal(pd.Series(s), df['col1'], check_names=False)