import numbers
from typing import AbstractSet, Any, cast, TYPE_CHECKING, TypeVar
from typing_extensions import Self
import sympy
from typing_extensions import Protocol
from cirq import study
from cirq._doc import doc_private
def parameter_names(val: Any) -> AbstractSet[str]:
    """Returns parameter names for this object.

    Args:
        val: Object for which to find the parameter names.
        check_symbols: If true, fall back to calling parameter_symbols.

    Returns:
        A set of parameter names if the object is parameterized. It the object
        does not implement the _parameter_names_ magic method or that method
        returns NotImplemented, returns an empty set.
    """
    if isinstance(val, sympy.Basic):
        return {cast(sympy.Symbol, symbol).name for symbol in val.free_symbols}
    if isinstance(val, numbers.Number):
        return set()
    if isinstance(val, (list, tuple)):
        return {name for e in val for name in parameter_names(e)}
    getter = getattr(val, '_parameter_names_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented:
        return result
    return set()