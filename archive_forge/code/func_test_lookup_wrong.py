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
def test_lookup_wrong(self, table_type, dtype):
    if dtype in (np.int8, np.uint8):
        N = 100
    else:
        N = 512
    table = table_type()
    keys = (np.arange(N) + N).astype(dtype)
    table.map_locations(keys)
    wrong_keys = np.arange(N).astype(dtype)
    result = table.lookup(wrong_keys)
    assert np.all(result == -1)