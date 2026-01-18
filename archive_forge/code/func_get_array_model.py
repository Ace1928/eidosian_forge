import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def get_array_model():

    def _trim_array_forward(model, X, is_train):

        def backprop(dY):
            return model.ops.alloc2f(dY.shape[0], dY.shape[1] + 1)
        return (X[:, :-1], backprop)
    return with_array2d(Model('trimarray', _trim_array_forward))