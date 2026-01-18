import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('m1,m2,expect_real', ((X, X, True), (X, Y, True), (X, H, True), (X, SQRT_X, False), (I, SQRT_Z, False)))
def test_hilbert_schmidt_inner_product_is_conjugate_symmetric(m1, m2, expect_real):
    v1 = cirq.hilbert_schmidt_inner_product(m1, m2)
    v2 = cirq.hilbert_schmidt_inner_product(m2, m1)
    assert v1 == v2.conjugate()
    assert np.isreal(v1) == expect_real
    if not expect_real:
        assert v1 != v2