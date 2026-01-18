from typing import Sequence, Tuple
import numpy as np
from cirq import linalg
def state_probabilities_by_indices(state_probability: np.ndarray, indices: Sequence[int], qid_shape: Tuple[int, ...]) -> np.ndarray:
    """Returns the probabilities for a state/measurement on the given indices.

    Args:
        state_probability: The multi-qubit state probability vector. This is an
            array of 2 to the power of the number of real numbers, and
            so state must be of size ``2**integer``.  The `state_probability` can be
            a vector of size ``2**integer`` or a tensor of shape
            ``(2, 2, ..., 2)``.
        indices: Which qubits are measured. The `state_probability` is assumed to be
            supplied in big endian order. That is the xth index of v, when
            expressed as a bitstring, has its largest values in the 0th index.
        qid_shape: The qid shape of the `state_probability`.

    Returns:
        State probabilities.
    """
    probs = state_probability.reshape((-1,))
    not_measured = [i for i in range(len(qid_shape)) if i not in indices]
    if linalg.can_numpy_support_shape(qid_shape):
        probs = probs.reshape(qid_shape)
        probs = np.transpose(probs, list(indices) + not_measured)
        probs = probs.reshape((-1,))
    else:
        probs = linalg.transpose_flattened_array(probs, qid_shape, list(indices) + not_measured)
    if len(not_measured):
        volume = np.prod([qid_shape[i] for i in indices])
        probs = probs.reshape((volume, -1))
        probs = np.sum(probs, axis=-1)
    return probs / np.sum(probs)