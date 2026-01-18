import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def binary_func_fixture():
    """
    Register a binary scalar function.
    """

    def binary_function(ctx, m, x):
        return pc.call_function('multiply', [m, x], memory_pool=ctx.memory_pool)
    func_name = 'y=mx'
    binary_doc = {'summary': 'y=mx', 'description': 'find y from y = mx'}
    pc.register_scalar_function(binary_function, func_name, binary_doc, {'m': pa.int64(), 'x': pa.int64()}, pa.int64())
    return (binary_function, func_name)