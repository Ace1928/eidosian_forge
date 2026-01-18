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
def test_mixed_column_names(self):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    for cols in [['あ', b'a'], [1, '2'], [1, 1.5]]:
        df.columns = pd.Index(cols, dtype=object)
        with pytest.warns(UserWarning):
            pa.Table.from_pandas(df)
        expected = df.copy()
        expected.columns = df.columns.values.astype(str)
        with pytest.warns(UserWarning):
            _check_pandas_roundtrip(df, expected=expected, preserve_index=True)