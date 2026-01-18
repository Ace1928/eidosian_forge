import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
@pytest.fixture
def ragged_input(ops, list_input):
    lengths = numpy.array([len(x) for x in list_input], dtype='i')
    if not list_input:
        return Ragged(ops.alloc2f(0, 0), lengths)
    else:
        return Ragged(ops.flatten(list_input), lengths)