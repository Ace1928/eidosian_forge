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
def test_map_array_dictionary_encoded(self):
    offsets = pa.array([0, 3, 5])
    items = pa.array(['a', 'b', 'c', 'a', 'd']).dictionary_encode()
    keys = pa.array(list(range(len(items))))
    arr = pa.MapArray.from_arrays(offsets, keys, items)
    expected = pd.Series([[(0, 'a'), (1, 'b'), (2, 'c')], [(3, 'a'), (4, 'd')]])
    actual = arr.to_pandas()
    tm.assert_series_equal(actual, expected, check_names=False)