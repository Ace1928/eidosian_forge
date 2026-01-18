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
def test_nested_types_from_ndarray_null_entries(self):
    s = pd.Series(np.array([np.nan, np.nan], dtype=object))
    for ty in [pa.list_(pa.int64()), pa.large_list(pa.int64()), pa.struct([pa.field('f0', 'int32')])]:
        result = pa.array(s, type=ty)
        expected = pa.array([None, None], type=ty)
        assert result.equals(expected)
        with pytest.raises(TypeError):
            pa.array(s.values, type=ty)