import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('mat', [np.diag([0]), np.diag([0.5]), np.diag([1, 0]), np.diag([0.5, 2]), np.array([[0, 1], [0, 0]])])
def test_bidiagonalize_unitary_fails(mat):
    with pytest.raises(ValueError):
        cirq.bidiagonalize_unitary_with_special_orthogonals(mat)