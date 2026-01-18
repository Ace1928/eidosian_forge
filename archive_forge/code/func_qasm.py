import string
from typing import TYPE_CHECKING, Union, Any, Tuple, TypeVar, Optional, Dict, Iterable
from typing_extensions import Protocol
from cirq import ops
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def qasm(val: Any, *, args: Optional[QasmArgs]=None, qubits: Optional[Iterable['cirq.Qid']]=None, default: TDefault=RaiseTypeErrorIfNotProvided) -> Union[str, TDefault]:
    """Returns QASM code for the given value, if possible.

    Different values require different sets of arguments. The general rule of
    thumb is that circuits don't need any, operations need a `QasmArgs`, and
    gates need both a `QasmArgs` and `qubits`.

    Args:
        val: The value to turn into QASM code.
        args: A `QasmArgs` object to pass into the value's `_qasm_` method.
            This is for needed for objects that only have a local idea of what's
            going on, e.g. a `cirq.Operation` in a bigger `cirq.Circuit`
            involving qubits that the operation wouldn't otherwise know about.
        qubits: A list of qubits that the value is being applied to. This is
            needed for `cirq.Gate` values, which otherwise wouldn't know what
            qubits to talk about.
        default: A default result to use if the value doesn't have a
            `_qasm_` method or that method returns `NotImplemented` or `None`.
            If not specified, non-decomposable values cause a `TypeError`.

    Returns:
        The result of `val._qasm_(...)`, if `val` has a `_qasm_`
        method and it didn't return `NotImplemented` or `None`. Otherwise
        `default` is returned, if it was specified. Otherwise an error is
        raised.

    Raises:
        TypeError: `val` didn't have a `_qasm_` method (or that method returned
            `NotImplemented` or `None`) and `default` wasn't set.
    """
    method = getattr(val, '_qasm_', None)
    result = NotImplemented
    if method is not None:
        kwargs: Dict[str, Any] = {}
        if args is not None:
            kwargs['args'] = args
        if qubits is not None:
            kwargs['qubits'] = tuple(qubits)
        result = method(**kwargs)
    if result is not None and result is not NotImplemented:
        return result
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if method is None:
        raise TypeError(f"object of type '{type(val)}' has no _qasm_ method.")
    raise TypeError(f"object of type '{type(val)}' does have a _qasm_ method, but it returned NotImplemented or None.")