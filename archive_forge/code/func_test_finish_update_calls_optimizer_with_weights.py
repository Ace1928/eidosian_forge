import numpy
import pytest
from hypothesis import given, settings
from mock import MagicMock
from numpy.testing import assert_allclose
from thinc.api import SGD, Dropout, Linear, chain
from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape
@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_finish_update_calls_optimizer_with_weights(W_b_input):
    model = get_model(W_b_input)
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    output, finish_update = model.begin_update(input_)
    seen_keys = set()

    def sgd(key, data, gradient, **kwargs):
        seen_keys.add(key)
        assert data.shape == gradient.shape
        return (data, gradient)
    grad_BO = numpy.ones((nr_batch, nr_out), dtype='f')
    grad_BI = finish_update(grad_BO)
    model.finish_update(sgd)
    for name in model.param_names:
        assert (model.id, name) in seen_keys