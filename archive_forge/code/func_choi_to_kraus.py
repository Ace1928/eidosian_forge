from typing import Sequence
import numpy as np
from cirq import protocols
def choi_to_kraus(choi: np.ndarray, atol: float=1e-10) -> Sequence[np.ndarray]:
    """Returns a Kraus representation of a channel with given Choi matrix.

    Quantum channel E: L(H1) -> L(H2) may be described by a collection of operators A_i, called
    Kraus operators, such that

        $$
        E(\\rho) = \\sum_i A_i \\rho A_i^\\dagger.
        $$

    Kraus representation is not unique. Alternatively, E may be specified by its Choi matrix J(E)
    defined as

        $$
        J(E) = (E \\otimes I)(|\\phi\\rangle\\langle\\phi|)
        $$

    where $|\\phi\\rangle = \\sum_i|i\\rangle|i\\rangle$ is the unnormalized maximally entangled state
    and I: L(H1) -> L(H1) is the identity map. Choi matrix is unique for a given channel.

    The most expensive step in the computation of a Kraus representation from a Choi matrix is
    the eigendecomposition of the Choi. Therefore, the cost of the conversion is O(d**6) where
    d is the dimension of the input and output Hilbert space.

    Args:
        choi: Choi matrix of the channel.
        atol: Tolerance used in checking if choi is positive and in deciding which Kraus
            operators to omit.

    Returns:
        Approximate Kraus representation of the quantum channel specified via a Choi matrix.
        Kraus operators with Frobenius norm smaller than atol are omitted.

    Raises:
        ValueError: when choi is not a positive square matrix.
    """
    d = int(np.round(np.sqrt(choi.shape[0])))
    if choi.shape != (d * d, d * d):
        raise ValueError(f'Invalid Choi matrix shape, expected {(d * d, d * d)}, got {choi.shape}')
    if not np.allclose(choi, choi.T.conj(), atol=atol):
        raise ValueError('Choi matrix must be Hermitian')
    w, v = np.linalg.eigh(choi)
    if np.any(w < -atol):
        raise ValueError(f'Choi matrix must be positive, got one with eigenvalues {w}')
    w = np.maximum(w, 0)
    u = np.sqrt(w) * v
    keep = np.linalg.norm(u.T, axis=-1) > atol
    return [k.reshape(d, d) for k, keep_i in zip(u.T, keep) if keep_i]