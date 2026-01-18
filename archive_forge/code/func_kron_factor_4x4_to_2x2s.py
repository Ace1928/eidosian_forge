import cmath
import math
from typing import (
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations
def kron_factor_4x4_to_2x2s(matrix: np.ndarray) -> Tuple[complex, np.ndarray, np.ndarray]:
    """Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

    Requires the matrix to be the kronecker product of two 2x2 unitaries.
    Requires the matrix to have a non-zero determinant.
    Giving an incorrect matrix will cause garbage output.

    Args:
        matrix: The 4x4 unitary matrix to factor.

    Returns:
        A scalar factor and a pair of 2x2 unit-determinant matrices. The
        kronecker product of all three is equal to the given matrix.

    Raises:
        ValueError:
            The given matrix can't be tensor-factored into 2x2 pieces.
    """
    a, b = max(((i, j) for i in range(4) for j in range(4)), key=lambda t: abs(matrix[t]))
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            f1[a >> 1 ^ i, b >> 1 ^ j] = matrix[a ^ i << 1, b ^ j << 1]
            f2[a & 1 ^ i, b & 1 ^ j] = matrix[a ^ i, b ^ j]
    f1 /= np.sqrt(np.linalg.det(f1)) or 1
    f2 /= np.sqrt(np.linalg.det(f2)) or 1
    g = matrix[a, b] / (f1[a >> 1, b >> 1] * f2[a & 1, b & 1])
    if np.real(g) < 0:
        f1 *= -1
        g = -g
    return (g, f1, f2)