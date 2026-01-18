import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def raising_func_fixture():
    """
    Register a scalar function which raises a custom exception.
    """

    def raising_func(ctx):
        raise MyError('error raised by scalar UDF')
    func_name = 'test_raise'
    doc = {'summary': 'raising function', 'description': ''}
    pc.register_scalar_function(raising_func, func_name, doc, {}, pa.int64())
    return (raising_func, func_name)