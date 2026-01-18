import numpy
import pytest
from thinc.api import NumpyOps, Ragged, registry, strings2arrays
from ..util import get_data_checker
@pytest.fixture
def array_data(ragged_data):
    return ragged_data.data