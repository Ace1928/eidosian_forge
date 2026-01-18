from typing import List, Optional, TYPE_CHECKING, Tuple, Sequence
import numpy as np
from cirq import linalg, value
from cirq.sim import simulation_utils
def measure_density_matrix(density_matrix: np.ndarray, indices: Sequence[int], qid_shape: Optional[Tuple[int, ...]]=None, out: Optional[np.ndarray]=None, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> Tuple[List[int], np.ndarray]:
    """Performs a measurement of the density matrix in the computational basis.

    This does not modify `density_matrix` unless the optional `out` is
    `density_matrix`.

    Args:
        density_matrix: The density matrix to be measured. This matrix is
            assumed to be positive semidefinite and trace one. The matrix is
            assumed to be of shape (2 ** integer, 2 ** integer) or
            (2, 2, ..., 2).
        indices: Which qubits are measured. The matrix is assumed to be supplied
            in big endian order. That is the xth index of v, when expressed as
            a bitstring, has the largest values in the 0th index.
        qid_shape: The qid shape of the density matrix.  Specify this argument
            when using qudits.
        out: An optional place to store the result. If `out` is the same as
            the `density_matrix` parameter, then `density_matrix` will be
            modified inline. If `out` is not None, then the result is put into
            `out`.  If `out` is None a new value will be allocated. In all of
            these cases `out` will be the same as the returned ndarray of the
            method. The shape and dtype of `out` will match that of
            `density_matrix` if `out` is None, otherwise it will match the
            shape and dtype of `out`.
        seed: A seed for the pseudorandom number generator.

    Returns:
        A tuple of a list and a numpy array. The list is an array of booleans
        corresponding to the measurement values (ordered by the indices). The
        numpy array is the post measurement matrix. This matrix has the same
        shape and dtype as the input matrix.

    Raises:
        ValueError if the dimension of the matrix is not compatible with a
            matrix of n qubits.
        IndexError if the indices are out of range for the number of qubits
            corresponding to the density matrix.
    """
    if qid_shape is None:
        num_qubits = _validate_num_qubits(density_matrix)
        qid_shape = (2,) * num_qubits
    else:
        _validate_density_matrix_qid_shape(density_matrix, qid_shape)
    meas_shape = _indices_shape(qid_shape, indices)
    arrout: np.ndarray
    if out is None:
        arrout = np.copy(density_matrix)
    elif out is density_matrix:
        arrout = density_matrix
    else:
        np.copyto(dst=out, src=density_matrix)
        arrout = out
    if len(indices) == 0:
        return ([], arrout)
    prng = value.parse_random_state(seed)
    initial_shape = density_matrix.shape
    probs = _probs(density_matrix, indices, qid_shape)
    result = prng.choice(len(probs), p=probs)
    measurement_bits = value.big_endian_int_to_digits(result, base=meas_shape)
    result_slice = linalg.slice_for_qubits_equal_to(indices, big_endian_qureg_value=result, qid_shape=qid_shape)
    mask = np.ones(qid_shape * 2, dtype=bool)
    mask[result_slice * 2] = False
    arrout.shape = qid_shape * 2
    arrout[mask] = 0
    arrout.shape = initial_shape
    arrout /= probs[result]
    return (measurement_bits, arrout)