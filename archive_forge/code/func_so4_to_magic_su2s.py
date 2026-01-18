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
def so4_to_magic_su2s(mat: np.ndarray, *, rtol: float=1e-05, atol: float=1e-08, check_preconditions: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """Finds 2x2 special-unitaries A, B where mat = Mag.H @ kron(A, B) @ Mag.

    Mag is the magic basis matrix:

        1  0  0  i
        0  i  1  0
        0  i -1  0     (times sqrt(0.5) to normalize)
        1  0  0 -i

    Args:
        mat: A real 4x4 orthogonal matrix.
        rtol: Per-matrix-entry relative tolerance on equality.
        atol: Per-matrix-entry absolute tolerance on equality.
        check_preconditions: When set, the code verifies that the given
            matrix is from SO(4). Defaults to set.

    Returns:
        A pair (A, B) of matrices in SU(2) such that Mag.H @ kron(A, B) @ Mag
        is approximately equal to the given matrix.

    Raises:
        ValueError: Bad matrix.
    """
    if check_preconditions:
        if mat.shape != (4, 4) or not predicates.is_special_orthogonal(mat, atol=atol, rtol=rtol):
            raise ValueError('mat must be 4x4 special orthogonal.')
    ab = combinators.dot(MAGIC, mat, MAGIC_CONJ_T)
    _, a, b = kron_factor_4x4_to_2x2s(ab)
    return (a, b)