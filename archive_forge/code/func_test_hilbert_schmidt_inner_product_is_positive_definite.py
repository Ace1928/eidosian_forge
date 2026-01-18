import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('m', (I, X, Y, Z, H, SQRT_X, SQRT_Y, SQRT_Z))
def test_hilbert_schmidt_inner_product_is_positive_definite(m):
    v = cirq.hilbert_schmidt_inner_product(m, m)
    assert np.isclose(np.imag(v), 1e-16)
    assert v.real > 0