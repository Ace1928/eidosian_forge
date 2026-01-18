import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def exception_agg_func_fixture():

    def func(ctx, x):
        raise RuntimeError('Oops')
        return pa.scalar(len(x))
    func_name = 'y=exception_len(x)'
    func_doc = empty_udf_doc
    pc.register_aggregate_function(func, func_name, func_doc, {'x': pa.int64()}, pa.int64())
    return (func, func_name)