import numpy
import pytest
from hypothesis import given
from thinc.api import Padded, Ragged, get_width
from thinc.types import ArgsKwargs
from thinc.util import (
from . import strategies
@pytest.mark.parametrize('obj,width', [(numpy.zeros((1, 2, 3, 4)), 4), (numpy.array(1), 0), (numpy.array([1, 2]), 3), ([numpy.zeros((1, 2)), numpy.zeros(1)], 2), (Ragged(numpy.zeros((1, 2)), numpy.zeros(1)), 2), (Padded(numpy.zeros((2, 1, 2)), numpy.zeros(2), numpy.array([1, 0]), numpy.array([0, 1])), 2), ([], 0)])
def test_get_width(obj, width):
    assert get_width(obj) == width