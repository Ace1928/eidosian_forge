import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
@pytest.fixture
def ragged_data_input(ragged_input):
    return (ragged_input.data, ragged_input.lengths)