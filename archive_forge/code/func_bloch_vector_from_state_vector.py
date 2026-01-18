from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def bloch_vector_from_state_vector(state_vector: np.ndarray, index: int, qid_shape: Optional[Tuple[int, ...]]=None) -> np.ndarray:
    """Returns the bloch vector of a qubit.

    Calculates the bloch vector of the qubit at index in the state vector,
    assuming state vector follows the standard Kronecker convention of
    numpy.kron.

    Args:
        state_vector: A sequence representing a state vector in which
            the ordering mapping to qubits follows the standard Kronecker
            convention of numpy.kron (big-endian).
        index: index of qubit who's bloch vector we want to find.
            follows the standard Kronecker convention of numpy.kron.
        qid_shape: specifies the dimensions of the qudits for the input
            `state_vector`.  If not specified, qubits are assumed and the
            `state_vector` must have a dimension a power of two.
            The qudit at `index` must be a qubit.

    Returns:
        A length 3 numpy array representing the qubit's bloch vector.

    Raises:
        ValueError: if the size of `state_vector `is not a power of 2 and the
            shape is not given or if the shape is given and `state_vector` has
            a size that contradicts this shape.
        IndexError: if index is out of range for the number of qubits or qudits
            corresponding to `state_vector`.
    """
    rho = density_matrix_from_state_vector(state_vector, [index], qid_shape=qid_shape)
    v = np.zeros(3, dtype=np.float32)
    v[0] = 2 * np.real(rho[0][1])
    v[1] = 2 * np.imag(rho[1][0])
    v[2] = np.real(rho[0][0] - rho[1][1])
    return v