import numpy
import pytest
from thinc.api import NumpyOps, Ragged, registry, strings2arrays
from ..util import get_data_checker
@pytest.fixture
def list_data(shapes):
    return [numpy.zeros(shape, dtype='f') for shape in shapes]