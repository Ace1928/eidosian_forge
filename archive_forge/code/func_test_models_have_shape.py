from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def test_models_have_shape(model1, model2, nI, nH, nO):
    assert model1.get_param('W').shape == (nH, nI)
    assert model1.get_param('b').shape == (nH,)
    assert model2.get_param('W').shape == (nO, nH)
    assert model2.get_param('b').shape == (nO,)