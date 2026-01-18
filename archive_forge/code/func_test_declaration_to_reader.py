import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.compute import field
def test_declaration_to_reader(table_source):
    with table_source.to_reader() as reader:
        assert reader.schema == pa.schema([('a', pa.int64()), ('b', pa.int64())])
        result = reader.read_all()
    expected = pa.table({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert result.equals(expected)