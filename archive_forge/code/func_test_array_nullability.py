import hypothesis as h
import pyarrow as pa
import pyarrow.tests.strategies as past
@h.given(past.arrays(past.primitive_types, nullable=False))
def test_array_nullability(array):
    assert array.null_count == 0