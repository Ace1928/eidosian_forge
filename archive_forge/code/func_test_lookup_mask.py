from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
def test_lookup_mask(self, table_type, dtype, writable):
    if table_type == ht.PyObjectHashTable:
        pytest.skip('Mask not supported for object')
    N = 3
    table = table_type(uses_mask=True)
    keys = (np.arange(N) + N).astype(dtype)
    mask = np.array([False, True, False])
    keys.flags.writeable = writable
    table.map_locations(keys, mask)
    result = table.lookup(keys, mask)
    expected = np.arange(N)
    tm.assert_numpy_array_equal(result.astype(np.int64), expected.astype(np.int64))
    result = table.lookup(np.array([1 + N]).astype(dtype), np.array([False]))
    tm.assert_numpy_array_equal(result.astype(np.int64), np.array([-1], dtype=np.int64))