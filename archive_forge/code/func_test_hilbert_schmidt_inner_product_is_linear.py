import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('a,m1,b,m2', ((1, X, 1, Z), (2, X, 3, Y), (2j, X, 3, I), (2, X, 3, X)))
def test_hilbert_schmidt_inner_product_is_linear(a, m1, b, m2):
    v1 = cirq.hilbert_schmidt_inner_product(H, a * m1 + b * m2)
    v2 = a * cirq.hilbert_schmidt_inner_product(H, m1) + b * cirq.hilbert_schmidt_inner_product(H, m2)
    assert v1 == v2