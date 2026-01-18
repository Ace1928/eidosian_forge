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
def test_get_set_contains_len(self, table_type, dtype):
    index = float('nan')
    table = table_type()
    assert index not in table
    table.set_item(index, 42)
    assert len(table) == 1
    assert index in table
    assert table.get_item(index) == 42
    table.set_item(index, 41)
    assert len(table) == 1
    assert index in table
    assert table.get_item(index) == 41