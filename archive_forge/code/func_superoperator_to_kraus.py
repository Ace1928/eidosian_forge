from typing import Sequence
import numpy as np
from cirq import protocols
def superoperator_to_kraus(superoperator: np.ndarray, atol: float=1e-10) -> Sequence[np.ndarray]:
    """Returns a Kraus representation of a channel specified via the superoperator matrix.

    Quantum channel E: L(H1) -> L(H2) may be described by a collection of operators A_i, called
    Kraus operators, such that

        $$
        E(\\rho) = \\sum_i A_i \\rho A_i^\\dagger.
        $$

    Kraus representation is not unique. Alternatively, E may be specified by its superoperator
    matrix K(E) defined so that

        $$
        K(E) vec(\\rho) = vec(E(\\rho))
        $$

    where the vectorization map $vec$ rearranges d-by-d matrices into d**2-dimensional vectors.
    Superoperator matrix is unique for a given channel. It is also called the natural
    representation of a quantum channel.

    The most expensive step in the computation of a Kraus representation from a superoperator
    matrix is eigendecomposition. Therefore, the cost of the conversion is O(d**6) where d is
    the dimension of the input and output Hilbert space.

    Args:
        superoperator: Superoperator matrix specifying a quantum channel.
        atol: Tolerance used to check which Kraus operators to omit.

    Returns:
        Sequence of Kraus operators of the channel specified by superoperator.
        Kraus operators with Frobenius norm smaller than atol are omitted.

    Raises:
        ValueError: If superoperator is not a valid superoperator matrix.
    """
    return choi_to_kraus(superoperator_to_choi(superoperator), atol=atol)