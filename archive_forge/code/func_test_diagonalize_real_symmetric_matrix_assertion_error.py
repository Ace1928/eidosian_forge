import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
def test_diagonalize_real_symmetric_matrix_assertion_error():
    with pytest.raises(AssertionError):
        matrix = np.array([[0.5, 0], [0, 1]])
        m = np.array([[0, 1], [0, 0]])
        p = cirq.diagonalize_real_symmetric_matrix(matrix)
        assert_diagonalized_by(m, p)