from typing import Sequence
import numpy as np
from cirq import protocols
def kraus_to_superoperator(kraus_operators: Sequence[np.ndarray]) -> np.ndarray:
    """Returns the matrix representation of the linear map with given Kraus operators.

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

    The computation of the superoperator matrix from a Kraus representation involves the sum of
    Kronecker products of all Kraus operators. This has the cost of O(kd**4) where k is the number
    of Kraus operators and d is the dimension of the input and output Hilbert space.

    Args:
        kraus_operators: Sequence of Kraus operators specifying a quantum channel.

    Returns:
        Superoperator matrix of the channel specified by kraus_operators.
    """
    d_out, d_in = kraus_operators[0].shape
    ops_arr = np.asarray(kraus_operators)
    m = np.einsum('bij,bkl->ikjl', ops_arr, ops_arr.conj())
    return m.reshape((d_out * d_out, d_in * d_in))