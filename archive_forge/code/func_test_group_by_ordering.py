import pytest
import pyarrow as pa
import pyarrow.compute as pc
from .test_extension_type import IntegerType
def test_group_by_ordering():
    table1 = pa.table({'a': [1, 2, 3, 4], 'b': ['a'] * 4})
    table2 = pa.table({'a': [1, 2, 3, 4], 'b': ['b'] * 4})
    table = pa.concat_tables([table1, table2])
    for _ in range(50):
        result = table.group_by('b', use_threads=False).aggregate([])
        assert result['b'] == pa.chunked_array([['a'], ['b']])