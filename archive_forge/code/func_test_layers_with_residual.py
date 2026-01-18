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
@pytest.mark.parametrize('name,kwargs,in_data,out_data', TEST_CASES_SUMMABLE)
def test_layers_with_residual(name, kwargs, in_data, out_data):
    cfg = {'@layers': 'residual.v1', 'layer': {'@layers': name, **kwargs}}
    model = registry.resolve({'config': cfg})['config']
    if 'LSTM' in name:
        model = with_padded(model)
    model.initialize(in_data, out_data)
    Y, backprop = model(in_data, is_train=True)
    assert_data_match(Y, out_data)
    dX = backprop(Y)
    assert_data_match(dX, in_data)