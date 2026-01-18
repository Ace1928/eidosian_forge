import numbers
from collections.abc import Iterable
from typing import Any, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.approximate_equality_protocol import approx_eq
def equal_up_to_global_phase(val: Any, other: Any, *, atol: Union[int, float]=1e-08) -> bool:
    """Determine whether two objects are equal up to global phase.

    If `val` implements a `_equal_up_to_global_phase_` method then it is
    invoked and takes precedence over all other checks:
     - For complex primitive type the magnitudes of the values are compared.
     - For `val` and `other` both iterable of the same length, consecutive
       elements are compared recursively. Types of `val` and `other` does not
       necessarily needs to match each other. They just need to be iterable and
       have the same structure.
     - For all other types, fall back to `_approx_eq_`

    Args:
        val: Source object for approximate comparison.
        other: Target object for approximate comparison.
        atol: The minimum absolute tolerance. This places an upper bound on
        the differences in *magnitudes* of two compared complex numbers.

    Returns:
        True if objects are approximately equal up to phase, False otherwise.
    """
    eq_up_to_phase_getter = getattr(val, '_equal_up_to_global_phase_', None)
    if eq_up_to_phase_getter is not None:
        result = eq_up_to_phase_getter(other, atol)
        if result is not NotImplemented:
            return result
    other_eq_up_to_phase_getter = getattr(other, '_equal_up_to_global_phase_', None)
    if other_eq_up_to_phase_getter is not None:
        result = other_eq_up_to_phase_getter(val, atol)
        if result is not NotImplemented:
            return result
    if isinstance(val, Iterable) and isinstance(other, Iterable):
        a = np.asarray(val)
        b = np.asarray(other)
        if a.dtype.kind in 'uifc' and b.dtype.kind in 'uifc':
            return linalg.allclose_up_to_global_phase(a, b, atol=atol)
    if isinstance(val, numbers.Number) and isinstance(other, numbers.Number):
        result = approx_eq(abs(val), abs(other), atol=atol)
        if result is not NotImplemented:
            return result
    return approx_eq(val, other, atol=atol)