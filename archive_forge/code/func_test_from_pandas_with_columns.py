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
def test_from_pandas_with_columns(self):
    df = pd.DataFrame({0: [1, 2, 3], 1: [1, 3, 3], 2: [2, 4, 5]}, columns=[1, 0])
    table = pa.Table.from_pandas(df, columns=[0, 1])
    expected = pa.Table.from_pandas(df[[0, 1]])
    assert expected.equals(table)
    record_batch_table = pa.RecordBatch.from_pandas(df, columns=[0, 1])
    record_batch_expected = pa.RecordBatch.from_pandas(df[[0, 1]])
    assert record_batch_expected.equals(record_batch_table)