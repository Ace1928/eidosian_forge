import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('a,b,match', [[X, Z, 'must be symmetric'], [Y, np.eye(2), 'must be real'], [np.eye(2), Y, 'must be real'], [np.eye(5), np.eye(4), 'shapes'], [e * cirq.testing.random_orthogonal(4) for e in random_bi_diagonalizable_pair(4)] + ['must be symmetric'], [np.array([[1, 1], [1, 0]]), np.array([[1, 1], [0, 1]]), 'mat1.T @ mat2 must be symmetric'], [np.array([[1, 1], [1, 0]]), np.array([[1, 0], [1, 1]]), 'mat1 @ mat2.T must be symmetric']])
def test_bidiagonalize_real_fails(a, b, match: str):
    a = np.array(a)
    b = np.array(b)
    with pytest.raises(ValueError, match=match):
        cirq.bidiagonalize_real_matrix_pair_with_symmetric_products(a, b)