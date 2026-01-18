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
def map_eigenvalues(matrix: np.ndarray, func: Callable[[complex], complex], *, atol: float=1e-08) -> np.ndarray:
    """Applies a function to the eigenvalues of a matrix.

    Given M = sum_k a_k |v_k><v_k|, returns f(M) = sum_k f(a_k) |v_k><v_k|.

    Args:
        matrix: The matrix to modify with the function.
        func: The function to apply to the eigenvalues of the matrix.
        rtol: Relative threshold used when separating eigenspaces.
        atol: Absolute threshold used when separating eigenspaces.

    Returns:
        The transformed matrix.
    """
    vals, vecs = unitary_eig(matrix, atol=atol)
    pieces = [np.outer(vec, np.conj(vec.T)) for vec in vecs.T]
    out_vals = np.vectorize(func)(vals.astype(complex))
    total = np.zeros(shape=matrix.shape)
    for piece, val in zip(pieces, out_vals):
        total = np.add(total, piece * val)
    return total