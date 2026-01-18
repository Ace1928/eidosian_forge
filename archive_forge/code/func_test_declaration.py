import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.compute import field
def test_declaration():
    table = pa.table({'a': [1, 2, 3], 'b': [4, 5, 6]})
    table_opts = TableSourceNodeOptions(table)
    filter_opts = FilterNodeOptions(field('a') > 1)
    decl = Declaration.from_sequence([Declaration('table_source', options=table_opts), Declaration('filter', options=filter_opts)])
    result = decl.to_table()
    assert result.equals(table.slice(1, 2))
    table_source = Declaration('table_source', options=table_opts)
    filtered = Declaration('filter', options=filter_opts, inputs=[table_source])
    result = filtered.to_table()
    assert result.equals(table.slice(1, 2))