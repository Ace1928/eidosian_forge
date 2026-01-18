from typing import List, Optional, TYPE_CHECKING, Tuple, Sequence
import numpy as np
from cirq import linalg, value
from cirq.sim import simulation_utils
def sample_density_matrix(density_matrix: np.ndarray, indices: Sequence[int], *, qid_shape: Optional[Tuple[int, ...]]=None, repetitions: int=1, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
    """Samples repeatedly from measurements in the computational basis.

    Note that this does not modify the density_matrix.

    Args:
        density_matrix: The density matrix to be measured. This matrix is
            assumed to be positive semidefinite and trace one. The matrix is
            assumed to be of shape (2 ** integer, 2 ** integer) or
            (2, 2, ..., 2).
        indices: Which qubits are measured. The density matrix rows and columns
            are assumed to be supplied in big endian order. That is the
            xth index of v, when expressed as a bitstring, has its largest
            values in the 0th index.
        qid_shape: The qid shape of the density matrix.  Specify this argument
            when using qudits.
        repetitions: The number of times to sample the density matrix.
        seed: A seed for the pseudorandom number generator.

    Returns:
        Measurement results with True corresponding to the ``|1⟩`` state.
        The outer list is for repetitions, and the inner corresponds to
        measurements ordered by the supplied qubits. These lists
        are wrapped as a numpy ndarray.

    Raises:
        ValueError: ``repetitions`` is less than one or size of ``matrix`` is
            not a power of 2.
        IndexError: An index from ``indices`` is out of range, given the number
            of qubits corresponding to the density matrix.
    """
    if repetitions < 0:
        raise ValueError(f'Number of repetitions cannot be negative. Was {repetitions}')
    if qid_shape is None:
        num_qubits = _validate_num_qubits(density_matrix)
        qid_shape = (2,) * num_qubits
    else:
        _validate_density_matrix_qid_shape(density_matrix, qid_shape)
    meas_shape = _indices_shape(qid_shape, indices)
    if repetitions == 0 or len(indices) == 0:
        return np.zeros(shape=(repetitions, len(indices)), dtype=np.int8)
    prng = value.parse_random_state(seed)
    probs = _probs(density_matrix, indices, qid_shape)
    result = prng.choice(len(probs), size=repetitions, p=probs)
    return np.array([value.big_endian_int_to_digits(result[i], base=meas_shape) for i in range(len(result))], dtype=np.int8)