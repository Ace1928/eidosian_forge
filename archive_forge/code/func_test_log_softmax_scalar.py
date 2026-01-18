import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.special as sc
def test_log_softmax_scalar():
    assert_allclose(sc.log_softmax(1.0), 0.0, rtol=1e-13)