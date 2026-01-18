import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_function_doc_validation():
    in_types = {'scalar': pa.int64()}
    out_type = pa.int64()
    func_doc = {'description': 'desc'}

    def add_const(ctx, scalar):
        return pc.call_function('add', [scalar, 1])
    with pytest.raises(ValueError, match='Function doc must contain a summary'):
        pc.register_scalar_function(add_const, 'test_no_summary', func_doc, in_types, out_type)
    func_doc = {'summary': 'test summary'}
    with pytest.raises(ValueError, match='Function doc must contain a description'):
        pc.register_scalar_function(add_const, 'test_no_desc', func_doc, in_types, out_type)