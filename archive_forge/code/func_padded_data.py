import numpy
import pytest
from thinc.api import NumpyOps, Ragged, registry, strings2arrays
from ..util import get_data_checker
@pytest.fixture
def padded_data(ops, list_data):
    return ops.list2padded(list_data)