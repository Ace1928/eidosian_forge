import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def test_matrix_multiply_by_1d_vector(self):

    def mul():
        np.mat(np.eye(2)) * np.ones(2)
    assert_raises(ValueError, mul)