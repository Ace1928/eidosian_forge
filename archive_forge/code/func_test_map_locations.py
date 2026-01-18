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
def test_map_locations(self, table_type, dtype):
    N = 10
    table = table_type()
    keys = np.full(N, np.nan, dtype=dtype)
    table.map_locations(keys)
    assert len(table) == 1
    assert table.get_item(np.nan) == N - 1