import numpy
import pytest
from thinc import registry
from thinc.api import (
@pytest.mark.parametrize('init_func', [glorot_uniform_init, zero_init, uniform_init, normal_init])
def test_initializer_func_setup(init_func):
    ops = NumpyOps()
    data = numpy.ndarray([1, 2, 3, 4], dtype='f')
    result = init_func(ops, data.shape)
    assert not numpy.array_equal(data, result)