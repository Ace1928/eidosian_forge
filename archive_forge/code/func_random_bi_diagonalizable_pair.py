import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
def random_bi_diagonalizable_pair(n: int, d1: Optional[int]=None, d2: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
    u = cirq.testing.random_orthogonal(n)
    s = random_real_diagonal_matrix(n, d1)
    z = random_real_diagonal_matrix(n, d2)
    v = cirq.testing.random_orthogonal(n)
    a = cirq.dot(u, s, v)
    b = cirq.dot(u, z, v)
    return (a, b)