import itertools
import dataclasses
import inspect
from collections import defaultdict
from typing import (
from typing_extensions import runtime_checkable
from typing_extensions import Protocol
from cirq import devices, ops
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def decompose_once(val: Any, default=RaiseTypeErrorIfNotProvided, *args, flatten: bool=True, context: Optional[DecompositionContext]=None, **kwargs):
    """Decomposes a value into operations, if possible.

    This method decomposes the value exactly once, instead of decomposing it
    and then continuing to decomposing the decomposed operations recursively
    until some criteria is met (which is what `cirq.decompose` does).

    Args:
        val: The value to call `_decompose_` on, if possible.
        default: A default result to use if the value doesn't have a
            `_decompose_` method or that method returns `NotImplemented` or
            `None`. If not specified, non-decomposable values cause a
            `TypeError`.
        *args: Positional arguments to forward into the `_decompose_` method of
            `val`.  For example, this is used to tell gates what qubits they are
            being applied to.
        flatten: If True, the returned OP-TREE will be flattened to a list of operations.
        context: Decomposition context specifying common configurable options for
            controlling the behavior of decompose.
        **kwargs: Keyword arguments to forward into the `_decompose_` method of
            `val`.

    Returns:
        The result of `val._decompose_(*args, **kwargs)`, if `val` has a
        `_decompose_` method and it didn't return `NotImplemented` or `None`.
        Otherwise `default` is returned, if it was specified. Otherwise an error
        is raised.

    Raises:
        TypeError: `val` didn't have a `_decompose_` method (or that method returned
            `NotImplemented` or `None`) and `default` wasn't set.
    """
    if context is None:
        context = DecompositionContext(ops.SimpleQubitManager(prefix=f'_decompose_protocol_{next(_CONTEXT_COUNTER)}'))
    method = getattr(val, '_decompose_with_context_', None)
    decomposed = NotImplemented if method is None else method(*args, **kwargs, context=context)
    if decomposed is NotImplemented or None:
        method = getattr(val, '_decompose_', None)
        decomposed = NotImplemented if method is None else method(*args, **kwargs)
    if decomposed is not NotImplemented and decomposed is not None:
        return list(ops.flatten_to_ops(decomposed)) if flatten else decomposed
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if method is None:
        raise TypeError(f"object of type '{type(val)}' has no _decompose_with_context_ or _decompose_ method.")
    raise TypeError(f'object of type {type(val)} does have a _decompose_ method, but it returned NotImplemented or None.')