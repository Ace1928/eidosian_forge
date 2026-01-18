import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.special as sc
@pytest.fixture
def log_softmax_x():
    x = np.arange(4)
    return x