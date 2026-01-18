import pytest
import pyarrow as pa
import pyarrow.compute as pc
from .test_extension_type import IntegerType
def test_complex_filter_table():
    t = pa.table({'a': [1, 2, 3, 4, 5, 6, 6], 'b': [10, 20, 30, 40, 50, 60, 61]})
    result = _filter_table(t, (pc.bit_wise_and(pc.field('a'), pc.scalar(1)) == pc.scalar(0)) & (pc.multiply(pc.field('a'), pc.scalar(10)) == pc.field('b')))
    assert result == pa.table({'a': [2, 4, 6], 'b': [20, 40, 60]})