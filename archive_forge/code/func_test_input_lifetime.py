import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_input_lifetime(unary_func_fixture):
    function, func_name = unary_func_fixture
    proxy_pool = pa.proxy_memory_pool(pa.default_memory_pool())
    assert proxy_pool.bytes_allocated() == 0
    v = pa.array([1] * 1000, type=pa.int64(), memory_pool=proxy_pool)
    assert proxy_pool.bytes_allocated() == 1000 * 8
    pc.call_function(func_name, [v])
    assert proxy_pool.bytes_allocated() == 1000 * 8
    v = None
    assert proxy_pool.bytes_allocated() == 0