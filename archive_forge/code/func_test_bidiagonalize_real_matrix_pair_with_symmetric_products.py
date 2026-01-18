import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('a,b', [(np.zeros((0, 0)), np.zeros((0, 0))), (np.eye(2), np.eye(2)), (np.eye(4), np.eye(4)), (np.eye(4), np.zeros((4, 4))), (H, H), (cirq.kron(np.eye(2), H), cirq.kron(H, np.eye(2))), (cirq.kron(np.eye(2), Z), cirq.kron(X, np.eye(2)))] + [random_bi_diagonalizable_pair(2) for _ in range(10)] + [random_bi_diagonalizable_pair(4) for _ in range(10)] + [random_bi_diagonalizable_pair(4, d1, d2) for _ in range(10) for d1 in range(4) for d2 in range(4)] + [random_bi_diagonalizable_pair(k) for k in range(1, 10)])
def test_bidiagonalize_real_matrix_pair_with_symmetric_products(a, b):
    a = np.array(a)
    b = np.array(b)
    p, q = cirq.bidiagonalize_real_matrix_pair_with_symmetric_products(a, b)
    assert_bidiagonalized_by(a, p, q)
    assert_bidiagonalized_by(b, p, q)