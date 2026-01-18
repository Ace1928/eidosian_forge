import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.compute import field
def test_aggregate_scalar(table_source):
    decl = Declaration.from_sequence([table_source, Declaration('aggregate', AggregateNodeOptions([('a', 'sum', None, 'a_sum')]))])
    result = decl.to_table()
    assert result.schema.names == ['a_sum']
    assert result['a_sum'].to_pylist() == [6]
    table = pa.table({'a': [1, 2, None]})
    aggr_opts = AggregateNodeOptions([('a', 'sum', pc.ScalarAggregateOptions(skip_nulls=False), 'a_sum')])
    decl = Declaration.from_sequence([Declaration('table_source', TableSourceNodeOptions(table)), Declaration('aggregate', aggr_opts)])
    result = decl.to_table()
    assert result.schema.names == ['a_sum']
    assert result['a_sum'].to_pylist() == [None]
    for target in ['a', field('a'), 0, field(0), ['a'], [field('a')], [0]]:
        aggr_opts = AggregateNodeOptions([(target, 'sum', None, 'a_sum')])
        decl = Declaration.from_sequence([table_source, Declaration('aggregate', aggr_opts)])
        result = decl.to_table()
        assert result.schema.names == ['a_sum']
        assert result['a_sum'].to_pylist() == [6]
    aggr_opts = AggregateNodeOptions([(['a', 'b'], 'sum', None, 'a_sum')])
    decl = Declaration.from_sequence([table_source, Declaration('aggregate', aggr_opts)])
    with pytest.raises(ValueError, match="Function 'sum' accepts 1 arguments but 2 passed"):
        _ = decl.to_table()
    aggr_opts = AggregateNodeOptions([('a', 'hash_sum', None, 'a_sum')])
    decl = Declaration.from_sequence([table_source, Declaration('aggregate', aggr_opts)])
    with pytest.raises(ValueError, match='is a hash aggregate function'):
        _ = decl.to_table()