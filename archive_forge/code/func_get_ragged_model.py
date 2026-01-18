import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def get_ragged_model():

    def _trim_ragged_forward(model, Xr, is_train):

        def backprop(dYr):
            dY = dYr.data
            dX = model.ops.alloc2f(dY.shape[0], dY.shape[1] + 1)
            return Ragged(dX, dYr.lengths)
        return (Ragged(Xr.data[:, :-1], Xr.lengths), backprop)
    return with_ragged(Model('trimragged', _trim_ragged_forward))