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
def kak_vector(unitary: Union[Iterable[np.ndarray], np.ndarray], *, rtol: float=1e-05, atol: float=1e-08, check_preconditions: bool=True) -> np.ndarray:
    """Compute the KAK vectors of one or more two qubit unitaries.

    Any 2 qubit unitary may be expressed as

    $$ U = k_l A k_r $$
    where $k_l, k_r$ are single qubit (local) unitaries and

    $$ A= \\exp \\left(i \\sum_{s=x,y,z} k_s \\sigma_{s}^{(0)} \\sigma_{s}^{(1)}
                 \\right) $$

    The vector entries are ordered such that
        $$ 0 ≤ |k_z| ≤ k_y ≤ k_x ≤ π/4 $$
    if $k_x$ = π/4, $k_z \\geq 0$.

    References:
        The appendix section of "Lower bounds on the complexity of simulating
        quantum gates".
        http://arxiv.org/abs/quant-ph/0307190v1

    Examples:
        >>> cirq.kak_vector(np.eye(4))
        array([0., 0., 0.])
        >>> unitaries = [cirq.unitary(cirq.CZ),cirq.unitary(cirq.ISWAP)]
        >>> cirq.kak_vector(unitaries) * 4 / np.pi
        array([[ 1.,  0., -0.],
               [ 1.,  1.,  0.]])

    Args:
        unitary: A unitary matrix, or a multi-dimensional array of unitary
            matrices. Must have shape (..., 4, 4), where the last two axes are
            for the unitary matrix and other axes are for broadcasting the kak
            vector computation.
        rtol: Per-matrix-entry relative tolerance on equality. Used in unitarity
            check of input.
        atol: Per-matrix-entry absolute tolerance on equality. Used in unitarity
            check of input. This also determines how close $k_x$ must be to π/4
            to guarantee $k_z$ ≥ 0. Must be non-negative.
        check_preconditions: When set to False, skips verifying that the input
            is unitary in order to increase performance.

    Returns:
        The KAK vector of the given unitary or unitaries. The output shape is
        the same as the input shape, except the two unitary matrix axes are
        replaced by the kak vector axis (i.e. the output has shape
        `unitary.shape[:-2] + (3,)`).

    Raises:
        ValueError: If `atol` is negative or if the unitary has the wrong shape.
    """
    unitary = np.asarray(unitary)
    if len(unitary) == 0:
        return np.zeros(shape=(0, 3), dtype=np.float64)
    if unitary.ndim < 2 or unitary.shape[-2:] != (4, 4):
        raise ValueError(f'Expected input unitary to have shape (...,4,4), but got {unitary.shape}.')
    if atol < 0:
        raise ValueError(f'Input atol must be positive, got {atol}.')
    if check_preconditions:
        actual = np.einsum('...ba,...bc', unitary.conj(), unitary) - np.eye(4)
        if not np.allclose(actual, np.zeros_like(actual), rtol=rtol, atol=atol):
            raise ValueError(f'Input must correspond to a 4x4 unitary matrix or tensor of unitary matrices. Received input:\n{unitary}')
    UB = np.einsum('...ab,...bc,...cd', MAGIC_CONJ_T, unitary, MAGIC)
    m = np.einsum('...ab,...cb', UB, UB)
    evals, _ = np.linalg.eig(m)
    phases = np.log(-1j * np.linalg.det(unitary)).imag + np.pi / 2
    evals *= np.exp(-1j * phases / 2)[..., np.newaxis]
    S2 = np.log(-1j * evals).imag + np.pi / 2
    S2 = np.sort(S2, axis=-1)[..., ::-1]
    n_shifted = np.round(S2.sum(axis=-1) / (2 * np.pi)).astype(int)
    for n in range(1, 5):
        S2[n_shifted == n, :n] -= 2 * np.pi
    S2[n_shifted == -1, :3] += 2 * np.pi
    k_vec = np.einsum('ab,...b', KAK_GAMMA, S2)[..., 1:] / 2
    return _canonicalize_kak_vector(k_vec, atol)