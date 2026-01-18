import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
def test_invalid_expression_ser_des():
    schema = pa.schema([pa.field('x', pa.int32()), pa.field('y', pa.int32())])
    expr = pc.equal(pc.field('x'), 7)
    bad_expr = pc.equal(pc.field('z'), 7)
    with pytest.raises(ValueError) as excinfo:
        pa.substrait.serialize_expressions([expr], [], schema)
    assert 'need to have the same length' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        pa.substrait.serialize_expressions([expr], ['foo', 'bar'], schema)
    assert 'need to have the same length' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        pa.substrait.serialize_expressions([bad_expr], ['expr'], schema)
    assert 'No match for FieldRef' in str(excinfo.value)