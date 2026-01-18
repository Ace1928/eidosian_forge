import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
def test_large_list_records():
    list_lengths = np.random.randint(0, 500, size=50)
    list_lengths[::10] = 0
    list_values = [list(map(int, np.random.randint(0, 100, size=x))) if i % 8 else None for i, x in enumerate(list_lengths)]
    a1 = pa.array(list_values)
    table = pa.Table.from_arrays([a1], ['int_lists'])
    _check_roundtrip(table)