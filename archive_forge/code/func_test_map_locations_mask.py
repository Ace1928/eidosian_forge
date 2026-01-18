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
def test_map_locations_mask(self, table_type, dtype, writable):
    if table_type == ht.PyObjectHashTable:
        pytest.skip('Mask not supported for object')
    N = 3
    table = table_type(uses_mask=True)
    keys = (np.arange(N) + N).astype(dtype)
    keys.flags.writeable = writable
    table.map_locations(keys, np.array([False, False, True]))
    for i in range(N - 1):
        assert table.get_item(keys[i]) == i
    with pytest.raises(KeyError, match=re.escape(str(keys[N - 1]))):
        table.get_item(keys[N - 1])
    assert table.get_na() == 2