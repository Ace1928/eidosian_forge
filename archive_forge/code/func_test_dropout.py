from typing import List, Optional
import numpy
import pytest
import srsly
from numpy.testing import assert_almost_equal
from thinc.api import Dropout, Model, NumpyOps, registry, with_padded
from thinc.backends import NumpyOps
from thinc.compat import has_torch
from thinc.types import Array2d, Floats2d, FloatsXd, Padded, Ragged, Shape
from thinc.util import data_validation, get_width
@pytest.mark.parametrize('data', [array2d, ragged, padded, [array2d, array2d]])
def test_dropout(data):
    model = Dropout(0.2)
    model.initialize(data, data)
    Y, backprop = model(data, is_train=False)
    assert_data_match(Y, data)
    dX = backprop(Y)
    assert_data_match(dX, data)