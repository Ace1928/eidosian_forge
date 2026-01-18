import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def sub_state_vector(state_vector: np.ndarray, keep_indices: List[int], *, default: np.ndarray=RaiseValueErrorIfNotProvided, atol: Union[int, float]=1e-06) -> np.ndarray:
    """Attempts to factor a state vector into two parts and return one of them.

    The input `state_vector` must have shape ``(2,) * n`` or ``(2 ** n)`` where
    `state_vector` is expressed over n qubits. The returned array will retain
    the same type of shape as the input state vector, either ``(2 ** k)`` or
    ``(2,) * k`` where k is the number of qubits kept.

    If a state vector $|\\psi\\rangle$ defined on n qubits is an outer product
    of kets like  $|\\psi\\rangle$ = $|x\\rangle \\otimes |y\\rangle$, and
    $|x\\rangle$ is defined over the subset ``keep_indices`` of k qubits, then
    this method will factor $|\\psi\\rangle$ into $|x\\rangle$ and $|y\\rangle$ and
    return $|x\\rangle$. Note that $|x\\rangle$ is not unique, because scalar
    multiplication may be absorbed by any factor of a tensor product,
    $e^{i \\theta} |y\\rangle \\otimes |x\\rangle =
    |y\\rangle \\otimes e^{i \\theta} |x\\rangle$

    This method randomizes the global phase of $|x\\rangle$ in order to avoid
    accidental reliance on the global phase being some specific value.

    If the provided `state_vector` cannot be factored into a pure state over
    `keep_indices`, the method will fall back to return `default`. If `default`
    is not provided, the method will fail and raise `ValueError`.

    Args:
        state_vector: The target state_vector.
        keep_indices: Which indices to attempt to get the separable part of the
            `state_vector` on.
        default: Determines the fallback behavior when `state_vector` doesn't
            have a pure state factorization. If the factored state is not pure
            and `default` is not set, a ValueError is raised. If default is set
            to a value, that value is returned.
        atol: The minimum tolerance for comparing the output state's coherence
            measure to 1.

    Returns:
        The state vector expressed over the desired subset of qubits.

    Raises:
        ValueError: If the `state_vector` is not of the correct shape or the
            indices are not a valid subset of the input `state_vector`'s
            indices.
        IndexError: If any indexes are out of range.
        EntangledStateError: If the result of factoring is not a pure state and
            `default` is not provided.

    """
    if not np.log2(state_vector.size).is_integer():
        raise ValueError(f'Input state_vector of size {state_vector.size} does not represent a state over qubits.')
    n_qubits = int(np.log2(state_vector.size))
    keep_dims = 1 << len(keep_indices)
    ret_shape: Union[Tuple[int], Tuple[int, ...]]
    if state_vector.shape == (state_vector.size,):
        ret_shape = (keep_dims,)
        state_vector = state_vector.reshape((2,) * n_qubits)
    elif state_vector.shape == (2,) * n_qubits:
        ret_shape = tuple((2 for _ in range(len(keep_indices))))
    else:
        raise ValueError('Input state_vector must be shaped like (2 ** n,) or (2,) * n')
    keep_dims = 1 << len(keep_indices)
    if not np.isclose(np.linalg.norm(state_vector), 1):
        raise ValueError('Input state must be normalized.')
    if len(set(keep_indices)) != len(keep_indices):
        raise ValueError(f'keep_indices were {keep_indices} but must be unique.')
    if any([ind >= n_qubits for ind in keep_indices]):
        raise ValueError('keep_indices {} are an invalid subset of the input state vector.')
    other_qubits = sorted(set(range(n_qubits)) - set(keep_indices))
    candidates = [state_vector[predicates.slice_for_qubits_equal_to(other_qubits, k)].reshape(keep_dims) for k in range(1 << len(other_qubits))]
    best_candidate = max(candidates, key=lambda c: float(np.linalg.norm(c, 2)))
    best_candidate = best_candidate / np.linalg.norm(best_candidate)
    left = np.conj(best_candidate.reshape((keep_dims,))).T
    coherence_measure = sum([abs(np.dot(left, c.reshape((keep_dims,)))) ** 2 for c in candidates])
    if protocols.approx_eq(coherence_measure, 1, atol=atol):
        return np.exp(2j * np.pi * np.random.random()) * best_candidate.reshape(ret_shape)
    if default is not RaiseValueErrorIfNotProvided:
        return default
    raise EntangledStateError(f'Input state vector could not be factored into pure state over indices {keep_indices}')