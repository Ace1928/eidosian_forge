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
def test_binary_column_name(self):
    if Version('2.0.0') <= Version(pd.__version__) < Version('2.3.0'):
        pytest.skip('Regression in pandas 2.0.0')
    column_data = ['い']
    key = 'あ'.encode()
    data = {key: column_data}
    df = pd.DataFrame(data)
    t = pa.Table.from_pandas(df, preserve_index=True)
    df2 = t.to_pandas()
    assert df.values[0] == df2.values[0]
    assert df.index.values[0] == df2.index.values[0]
    assert df.columns[0] == key