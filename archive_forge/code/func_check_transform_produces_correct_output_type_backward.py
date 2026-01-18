import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def check_transform_produces_correct_output_type_backward(model, inputs, checker):
    outputs, backprop = model.begin_update(inputs)
    d_inputs = backprop(outputs)
    assert checker(inputs, d_inputs)