import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from numpy.testing import assert_allclose, assert_equal
def test_lsmr_output_shape():
    x = splin.lsmr(A=np.ones((10, 1)), b=np.zeros(10), x0=np.ones(1))[0]
    assert_equal(x.shape, (1,))