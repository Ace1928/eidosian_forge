import numpy
import pytest
from hypothesis import given, settings
from mock import MagicMock
from numpy.testing import assert_allclose
from thinc.api import SGD, Dropout, Linear, chain
from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape
@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_dropout_gives_zero_gradients(W_b_input):
    model = chain(get_model(W_b_input), Dropout(1.0))
    nr_batch, nr_out, nr_in = get_shape(W_b_input)
    W, b, input_ = W_b_input
    for node in model.walk():
        if node.name == 'dropout':
            node.attrs['dropout_rate'] = 1.0
    fwd_dropped, finish_update = model.begin_update(input_)
    grad_BO = numpy.ones((nr_batch, nr_out), dtype='f')
    grad_BI = finish_update(grad_BO)
    assert all((val == 0.0 for val in grad_BI.flatten()))