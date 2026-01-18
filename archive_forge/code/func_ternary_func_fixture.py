import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def ternary_func_fixture():
    """
    Register a ternary scalar function.
    """

    def ternary_function(ctx, m, x, c):
        mx = pc.call_function('multiply', [m, x], memory_pool=ctx.memory_pool)
        return pc.call_function('add', [mx, c], memory_pool=ctx.memory_pool)
    ternary_doc = {'summary': 'y=mx+c', 'description': 'find y from y = mx + c'}
    func_name = 'y=mx+c'
    pc.register_scalar_function(ternary_function, func_name, ternary_doc, {'array1': pa.int64(), 'array2': pa.int64(), 'array3': pa.int64()}, pa.int64())
    return (ternary_function, func_name)