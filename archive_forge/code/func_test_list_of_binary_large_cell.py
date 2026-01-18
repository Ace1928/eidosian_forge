import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.slow
@pytest.mark.pandas
@pytest.mark.large_memory
def test_list_of_binary_large_cell():
    data = []
    data.extend([[b'x' * 1000000] * 10] * 214)
    arr = pa.array(data)
    table = pa.Table.from_arrays([arr], ['chunky_cells'])
    read_table = _simple_table_roundtrip(table)
    assert table.equals(read_table)