import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
@pytest.fixture
def padded_data_input(padded_input):
    x = padded_input
    return (x.data, x.size_at_t, x.lengths, x.indices)