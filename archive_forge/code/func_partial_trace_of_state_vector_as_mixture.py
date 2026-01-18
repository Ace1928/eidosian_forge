import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def partial_trace_of_state_vector_as_mixture(state_vector: np.ndarray, keep_indices: List[int], *, atol: Union[int, float]=1e-08) -> Tuple[Tuple[float, np.ndarray], ...]:
    """Returns a mixture representing a state vector with only some qubits kept.

    The input state vector can have any shape, but if it is one-dimensional it
    will be interpreted as qubits, since that is the most common case, and fail
    if the dimension is not size `2 ** n`. States in the output mixture will
    retain the same type of shape as the input state vector.

    If the state vector cannot be factored into a pure state over `keep_indices`
    then eigendecomposition is used and the output mixture will not be unique.

    Args:
        state_vector: The state vector to take the partial trace over.
        keep_indices: Which indices to take the partial trace of the
            state_vector on.
        atol: The tolerance for determining that a factored state is pure.

    Returns:
        A single-component mixture in which the factored state vector has
        probability '1' if the partially traced state is pure, or else a
        mixture of the default eigendecomposition of the mixed state's
        partial trace.

    Raises:
        ValueError: If the input `state_vector` is one dimension, but that
            dimension size is not a power of two.
        IndexError: If any indexes are out of range.
    """
    if state_vector.ndim == 1:
        dims = int(np.log2(state_vector.size))
        if 2 ** dims != state_vector.size:
            raise ValueError(f'Cannot infer underlying shape of {state_vector.shape}.')
        state_vector = state_vector.reshape((2,) * dims)
        ret_shape: Tuple[int, ...] = (2 ** len(keep_indices),)
    else:
        ret_shape = tuple((state_vector.shape[i] for i in keep_indices))
    try:
        state, _ = factor_state_vector(state_vector, keep_indices, atol=atol)
        return ((1.0, state.reshape(ret_shape)),)
    except EntangledStateError:
        pass
    rho = np.outer(state_vector, np.conj(state_vector)).reshape(state_vector.shape * 2)
    keep_rho = partial_trace(rho, keep_indices).reshape((np.prod(ret_shape),) * 2)
    eigvals, eigvecs = np.linalg.eigh(keep_rho)
    mixture = tuple(zip(eigvals, [vec.reshape(ret_shape) for vec in eigvecs.T]))
    return tuple([(float(p[0]), p[1]) for p in mixture if not protocols.approx_eq(p[0], 0.0)])