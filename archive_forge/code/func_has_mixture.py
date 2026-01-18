from typing import Any, Sequence, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.protocols.has_unitary_protocol import has_unitary
from cirq.type_workarounds import NotImplementedType
def has_mixture(val: Any, *, allow_decompose: bool=True) -> bool:
    """Returns whether the value has a mixture representation.

    Args:
        val: The value to check.
        allow_decompose: Used by internal methods to stop redundant
            decompositions from being performed (e.g. there's no need to
            decompose an object to check if it is unitary as part of determining
            if the object is a quantum channel, when the quantum channel check
            will already be doing a more general decomposition check). Defaults
            to True. When false, the decomposition strategy for determining
            the result is skipped.

    Returns:
        If `val` has a `_has_mixture_` method and its result is not
        NotImplemented, that result is returned. Otherwise, if the value
        has a `_mixture_` method return True if that has a non-default value.
        Returns False if neither function exists.
    """
    mixture_getter = getattr(val, '_has_mixture_', None)
    result = NotImplemented if mixture_getter is None else mixture_getter()
    if result is not NotImplemented:
        return result
    if has_unitary(val, allow_decompose=False):
        return True
    if allow_decompose:
        operations, _, _ = _try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return all((has_mixture(val) for val in operations))
    return mixture(val, None) is not None