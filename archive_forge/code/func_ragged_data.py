import numpy
import pytest
from thinc.api import NumpyOps, Ragged, registry, strings2arrays
from ..util import get_data_checker
@pytest.fixture
def ragged_data(ops, list_data):
    lengths = numpy.array([len(x) for x in list_data], dtype='i')
    if not list_data:
        return Ragged(ops.alloc2f(0, 0), lengths)
    else:
        return Ragged(ops.flatten(list_data), lengths)