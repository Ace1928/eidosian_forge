from typing import Any, Sequence, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.protocols.has_unitary_protocol import has_unitary
from cirq.type_workarounds import NotImplementedType
def mixture(val: Any, default: Any=RaiseTypeErrorIfNotProvided) -> Sequence[Tuple[float, np.ndarray]]:
    """Return a sequence of tuples representing a probabilistic unitary.

    A mixture is described by an iterable of tuples of the form

        (probability of unitary, unitary as numpy array)

    The probability components of the tuples must sum to 1.0 and be
    non-negative.

    Args:
        val: The value to decompose into a mixture of unitaries.
        default: A default value if val does not support mixture.

    Returns:
        An iterable of tuples of size 2. The first element of the tuple is a
        probability (between 0 and 1) and the second is the object that occurs
        with that probability in the mixture. The probabilities will sum to 1.0.

    Raises:
        TypeError: If `val` has no `_mixture_` or `_unitary_` mehod, or if it
            does and this method returned `NotImplemented`.
    """
    mixture_getter = getattr(val, '_mixture_', None)
    result = NotImplemented if mixture_getter is None else mixture_getter()
    if result is not NotImplemented:
        return result
    unitary_getter = getattr(val, '_unitary_', None)
    result = NotImplemented if unitary_getter is None else unitary_getter()
    if result is not NotImplemented:
        return ((1.0, result),)
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if mixture_getter is None and unitary_getter is None:
        raise TypeError(f"object of type '{type(val)}' has no _mixture_ or _unitary_ method.")
    raise TypeError(f"object of type '{type(val)}' does have a _mixture_ or _unitary_ method, but it returned NotImplemented.")