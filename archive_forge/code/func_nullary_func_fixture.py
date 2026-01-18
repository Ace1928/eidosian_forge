import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def nullary_func_fixture():
    """
    Register a nullary scalar function.
    """

    def nullary_func(context):
        return pa.array([42] * context.batch_length, type=pa.int64(), memory_pool=context.memory_pool)
    func_doc = {'summary': 'random function', 'description': 'generates a random value'}
    func_name = 'test_nullary_func'
    pc.register_scalar_function(nullary_func, func_name, func_doc, {}, pa.int64())
    return (nullary_func, func_name)