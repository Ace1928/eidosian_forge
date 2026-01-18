import numpy
import pytest
from thinc.api import NumpyOps, Ragged, registry, strings2arrays
from ..util import get_data_checker
def test_list2ragged(list_data, ragged_data):
    check_transform('list2ragged.v1', list_data, ragged_data)