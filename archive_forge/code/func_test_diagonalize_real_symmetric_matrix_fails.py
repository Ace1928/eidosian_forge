import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('matrix', [np.array([[0, 1], [0, 0]]), np.array([[1, 1], [0, 1]]), np.array([[1, 1j], [-1j, 1]]), np.array([[1, 1j], [1j, 1]]), np.array([[3, 1], [7, 3]])])
def test_diagonalize_real_symmetric_matrix_fails(matrix):
    with pytest.raises(ValueError):
        _ = cirq.diagonalize_real_symmetric_matrix(matrix)