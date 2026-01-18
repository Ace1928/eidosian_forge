from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator, Stinespring
from .dihedral.random import random_cnotdihedral
from .symplectic.random import random_clifford, random_pauli, random_pauli_list
def random_hermitian(dims: int | tuple, traceless: bool=False, seed: int | np.random.Generator | None=None):
    """Return a random hermitian Operator.

    The operator is sampled from Gaussian Unitary Ensemble.

    Args:
        dims (int or tuple): the input dimension of the Operator.
        traceless (bool): Optional. If True subtract diagonal entries to
                          return a traceless hermitian operator
                          (Default: False).
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Operator: a Hermitian operator.
    """
    if seed is None:
        rng = DEFAULT_RNG
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)
    dim = np.prod(dims)
    from scipy import stats
    if traceless:
        mat = np.zeros((dim, dim), dtype=complex)
    else:
        mat = np.diag(stats.norm.rvs(scale=1, size=dim, random_state=rng).astype(complex))
    num_tril = dim * (dim - 1) // 2
    real_tril = stats.norm.rvs(scale=0.5, size=num_tril, random_state=rng)
    imag_tril = stats.norm.rvs(scale=0.5, size=num_tril, random_state=rng)
    rows, cols = np.tril_indices(dim, -1)
    mat[rows, cols] = real_tril + 1j * imag_tril
    mat[cols, rows] = real_tril - 1j * imag_tril
    return Operator(mat, input_dims=dims, output_dims=dims)