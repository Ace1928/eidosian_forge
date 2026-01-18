from typing import Sequence
import numpy as np
from cirq import protocols
def choi_to_superoperator(choi: np.ndarray) -> np.ndarray:
    """Returns the superoperator matrix of a quantum channel specified via the Choi matrix.

    Quantum channel E: L(H1) -> L(H2) may be specified by its Choi matrix J(E) defined as

        $$
        J(E) = (E \\otimes I)(|\\phi\\rangle\\langle\\phi|)
        $$

    where $|\\phi\\rangle = \\sum_i|i\\rangle|i\\rangle$ is the unnormalized maximally entangled state
    and I: L(H1) -> L(H1) is the identity map. Choi matrix is unique for a given channel.
    Alternatively, E may be specified by its superoperator matrix K(E) defined so that

        $$
        K(E) vec(\\rho) = vec(E(\\rho))
        $$

    where the vectorization map $vec$ rearranges d-by-d matrices into d**2-dimensional vectors.
    Superoperator matrix is unique for a given channel. It is also called the natural
    representation of a quantum channel.

    A quantum channel can be viewed as a tensor with four indices. Different ways of grouping
    the indices into two pairs yield different matrix representations of the channel, including
    the superoperator and Choi representations. Hence, the conversion between the superoperator
    and Choi matrices is a permutation of matrix elements effected by reshaping the array and
    swapping its axes. Therefore, its cost is O(d**4) where d is the dimension of the input and
    output Hilbert space.

    Args:
        choi: Choi matrix specifying a quantum channel.

    Returns:
        Superoperator matrix of the channel specified by choi.

    Raises:
        ValueError: If Choi is not Hermitian or is of invalid shape.
    """
    d = int(np.round(np.sqrt(choi.shape[0])))
    if choi.shape != (d * d, d * d):
        raise ValueError(f'Invalid Choi matrix shape, expected {(d * d, d * d)}, got {choi.shape}')
    if not np.allclose(choi, choi.T.conj()):
        raise ValueError('Choi matrix must be Hermitian')
    c = np.reshape(choi, (d, d, d, d))
    s = np.swapaxes(c, 1, 2)
    return np.reshape(s, (d * d, d * d))