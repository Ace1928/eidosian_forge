import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.compute import field
def test_aggregate_hash():
    table = pa.table({'a': [1, 2, None], 'b': ['foo', 'bar', 'foo']})
    table_opts = TableSourceNodeOptions(table)
    table_source = Declaration('table_source', options=table_opts)
    aggr_opts = AggregateNodeOptions([('a', 'hash_count', None, 'count(a)')], keys=['b'])
    decl = Declaration.from_sequence([table_source, Declaration('aggregate', aggr_opts)])
    result = decl.to_table()
    expected = pa.table({'b': ['foo', 'bar'], 'count(a)': [1, 1]})
    assert result.equals(expected)
    aggr_opts = AggregateNodeOptions([('a', 'hash_count', pc.CountOptions('all'), 'count(a)')], keys=['b'])
    decl = Declaration.from_sequence([table_source, Declaration('aggregate', aggr_opts)])
    result = decl.to_table()
    expected_all = pa.table({'b': ['foo', 'bar'], 'count(a)': [2, 1]})
    assert result.equals(expected_all)
    aggr_opts = AggregateNodeOptions([('a', 'hash_count', None, 'count(a)')], keys=[field('b')])
    decl = Declaration.from_sequence([table_source, Declaration('aggregate', aggr_opts)])
    result = decl.to_table()
    assert result.equals(expected)
    aggr_opts = AggregateNodeOptions([('a', 'sum', None, 'a_sum')], keys=['b'])
    decl = Declaration.from_sequence([table_source, Declaration('aggregate', aggr_opts)])
    with pytest.raises(ValueError):
        _ = decl.to_table()