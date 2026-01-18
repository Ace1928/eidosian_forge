import numpy as np
from scipy.linalg import bandwidth, issymmetric, ishermitian
import pytest
from pytest import raises
def test_bandwidth_non2d_input():
    A = np.array([1, 2, 3])
    raises(ValueError, bandwidth, A)
    A = np.array([[[1, 2, 3], [4, 5, 6]]])
    raises(ValueError, bandwidth, A)