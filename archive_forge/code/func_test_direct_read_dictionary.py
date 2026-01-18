import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_direct_read_dictionary():
    repeats = 10
    nunique = 5
    data = [[util.rands(10) for i in range(nunique)] * repeats]
    table = pa.table(data, names=['f0'])
    bio = pa.BufferOutputStream()
    pq.write_table(table, bio)
    contents = bio.getvalue()
    result = pq.read_table(pa.BufferReader(contents), read_dictionary=['f0'])
    expected = pa.table([table[0].dictionary_encode()], names=['f0'])
    assert result.equals(expected)