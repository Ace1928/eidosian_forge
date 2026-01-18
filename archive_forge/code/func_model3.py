import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
@pytest.fixture
def model3(nO):
    return Linear(nO, nO)