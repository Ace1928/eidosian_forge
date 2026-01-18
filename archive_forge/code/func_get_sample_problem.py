import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_allclose, assert_
from scipy.sparse.linalg._isolve import minres
from pytest import raises as assert_raises
def get_sample_problem():
    np.random.seed(1234)
    matrix = np.random.rand(10, 10)
    matrix = matrix + matrix.T
    vector = np.random.rand(10)
    return (matrix, vector)