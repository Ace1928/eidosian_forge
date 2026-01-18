import numpy as np
from scipy.linalg import bandwidth, issymmetric, ishermitian
import pytest
from pytest import raises
def test_ishermitian_complex_decimals():
    A = np.arange(1, 10).astype(complex).reshape(3, 3)
    A += np.arange(-4, 5).astype(complex).reshape(3, 3) * 1j
    A /= np.pi
    A = A + A.T.conj()
    assert ishermitian(A)