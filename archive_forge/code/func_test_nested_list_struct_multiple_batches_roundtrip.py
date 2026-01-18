import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
def test_nested_list_struct_multiple_batches_roundtrip(tempdir):
    data = [[{'x': 'abc', 'y': 'abc'}]] * 100 + [[{'x': 'abc', 'y': 'gcb'}]] * 100
    table = pa.table([pa.array(data)], names=['column'])
    _check_roundtrip(table, row_group_size=20)
    data = pa.array([{'a': '1', 'b': '2'}, {'a': '3', 'b': '4'}, {'a': '5', 'b': '6'}] * 10)
    table = pa.table({'column': data})
    _check_roundtrip(table, row_group_size=10)