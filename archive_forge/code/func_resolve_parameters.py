import numbers
from typing import AbstractSet, Any, cast, TYPE_CHECKING, TypeVar
from typing_extensions import Self
import sympy
from typing_extensions import Protocol
from cirq import study
from cirq._doc import doc_private
def resolve_parameters(val: T, param_resolver: 'cirq.ParamResolverOrSimilarType', recursive: bool=True) -> T:
    """Resolves symbol parameters in the effect using the param resolver.

    This function will use the `_resolve_parameters_` magic method
    of `val` to resolve any Symbols with concrete values from the given
    parameter resolver.

    Args:
        val: The object to resolve (e.g. the gate, operation, etc)
        param_resolver: the object to use for resolving all symbols
        recursive: if True, resolves parameters recursively over the
            resolver; otherwise performs a single resolution step.

    Returns:
        a gate or operation of the same type, but with all Symbols
        replaced with floats or terminal symbols according to the
        given `cirq.ParamResolver`. If `val` has no `_resolve_parameters_`
        method or if it returns NotImplemented, `val` itself is returned.
        Note that in some cases, such as when directly resolving a sympy
        Symbol, the return type could differ from the input type; however,
        for the much more common case of resolving parameters on cirq
        objects (or if resolving a Union[Symbol, float] instead of just a
        Symbol), the return type will be the same as val so we reflect
        that in the type signature of this protocol function.

    Raises:
        RecursionError if the ParamResolver detects a loop in resolution.
        ValueError if `recursive=False` is passed to an external
            _resolve_parameters_ method with no `recursive` parameter.
    """
    if not param_resolver:
        return val
    param_resolver = study.ParamResolver(param_resolver)
    if isinstance(val, sympy.Expr):
        return cast(T, param_resolver.value_of(val, recursive))
    if isinstance(val, (list, tuple)):
        return cast(T, type(val)((resolve_parameters(e, param_resolver, recursive) for e in val)))
    is_parameterized = getattr(val, '_is_parameterized_', None)
    if is_parameterized is not None and (not is_parameterized()):
        return val
    getter = getattr(val, '_resolve_parameters_', None)
    if getter is None:
        result = NotImplemented
    else:
        result = getter(param_resolver, recursive)
    if result is not NotImplemented:
        return result
    else:
        return val