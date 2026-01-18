import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def check_transform_doesnt_change_noop_values(model, inputs, d_outputs):
    outputs, backprop = model.begin_update(inputs)
    d_inputs = backprop(d_outputs)
    if isinstance(outputs, list):
        for i in range(len(outputs)):
            numpy.testing.assert_equal(inputs[i], outputs[i])
            numpy.testing.assert_equal(d_outputs[i], d_inputs[i])
    elif isinstance(outputs, numpy.ndarray):
        numpy.testing.assert_equal(inputs, outputs)
        numpy.testing.assert_equal(d_outputs, d_inputs)
    elif isinstance(outputs, Ragged):
        numpy.testing.assert_equal(inputs.data, outputs.data)
        numpy.testing.assert_equal(d_outputs.data, d_inputs.data)
    elif isinstance(outputs, Padded):
        numpy.testing.assert_equal(inputs.data, outputs.data)
        numpy.testing.assert_equal(d_inputs.data, d_inputs.data)