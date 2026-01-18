import numpy
import pytest
from hypothesis import given, settings
from mock import MagicMock
from numpy.testing import assert_allclose
from thinc.api import SGD, Dropout, Linear, chain
from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape
def sgd(key, data, gradient, **kwargs):
    seen_keys.add(key)
    assert data.shape == gradient.shape
    return (data, gradient)