import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_add_edge_cases():
    data = numpy.asarray([[1, 2, 3, 4]], dtype='f')
    with pytest.raises(TypeError):
        add()
    model = add(Linear(), Linear())
    model._layers = []
    Y, backprop = model(data, is_train=True)
    assert numpy.array_equal(data, Y)
    dX = backprop(Y)
    assert numpy.array_equal(dX, data)