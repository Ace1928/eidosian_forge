from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def quantum_state(state: 'cirq.QUANTUM_STATE_LIKE', qid_shape: Optional[Tuple[int, ...]]=None, *, copy: bool=False, validate: bool=True, dtype: Optional['DTypeLike']=None, atol: float=1e-07) -> QuantumState:
    """Create a QuantumState object from a state-like object.

    Args:
        state: The state-like object.
        qid_shape: The qid shape.
        copy: Whether to copy the data underlying the state.
        validate: Whether to check if the given data and qid shape
            represent a valid quantum state with the given dtype.
        dtype: The desired data type.
        atol: Absolute numerical tolerance to use for validation.

    Raises:
        ValueError: Invalid quantum state.
        ValueError: The qid shape was not specified and could not be inferred.
    """
    if isinstance(state, QuantumState):
        if qid_shape is not None and state.qid_shape != qid_shape:
            raise ValueError(f'The specified qid shape must be the same as the qid shape of the given state.\nSpecified shape: {qid_shape}\nShape of state: {state.qid_shape}.')
        if copy or (dtype and dtype != state.dtype):
            if dtype and dtype != state.dtype:
                data = state.data.astype(dtype, casting='unsafe', copy=True)
            else:
                data = state.data.copy()
            new_state = QuantumState(data, state.qid_shape)
        else:
            new_state = state
        if validate:
            new_state.validate(dtype=dtype, atol=atol)
        return new_state
    if isinstance(state, value.ProductState):
        actual_qid_shape = (2,) * len(state)
        if qid_shape is not None and qid_shape != actual_qid_shape:
            raise ValueError(f'The specified qid shape must be the same as the qid shape of the given state.\nSpecified shape: {qid_shape}\nShape of state: {actual_qid_shape}.')
        if dtype is None:
            dtype = DEFAULT_COMPLEX_DTYPE
        data = state.state_vector().astype(dtype, casting='unsafe', copy=False)
        qid_shape = actual_qid_shape
    elif isinstance(state, int):
        if qid_shape is None:
            raise ValueError('The qid shape of the given state is ambiguous. Please specify the qid shape explicitly using the qid_shape argument.')
        dim = np.prod(qid_shape, dtype=np.int64).item()
        if not 0 <= state < dim:
            raise ValueError(f'Computational basis state is out of range.\n\nstate={state!r}\nMIN_STATE=0\nMAX_STATE=product(qid_shape)-1={dim - 1}\nqid_shape={qid_shape!r}\n')
        if dtype is None:
            dtype = DEFAULT_COMPLEX_DTYPE
        data = one_hot(index=state, shape=(dim,), dtype=dtype)
    else:
        data = np.array(state, copy=False)
        if qid_shape is None:
            qid_shape = infer_qid_shape(state)
        if data.ndim == 1 and data.dtype.kind != 'c':
            if len(qid_shape) == np.prod(qid_shape, dtype=np.int64):
                raise ValueError('Because len(qid_shape) == product(qid_shape), it is ambiguous whether the given state contains state vector amplitudes or per-qudit computational basis values. In this situation you are required to pass in a state vector that is a numpy array with a complex dtype.')
            if data.shape == (len(qid_shape),):
                data = _qudit_values_to_state_tensor(state_vector=data, qid_shape=qid_shape, dtype=dtype)
        if copy or (dtype and dtype != data.dtype):
            if dtype and dtype != data.dtype:
                data = data.astype(dtype, casting='unsafe', copy=True)
            else:
                data = data.copy()
    return QuantumState(data=data, qid_shape=qid_shape, validate=validate, dtype=dtype, atol=atol)