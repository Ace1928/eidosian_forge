from typing import Any, FrozenSet, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import measurement_key_protocol
from cirq.type_workarounds import NotImplementedType
def control_keys(val: Any) -> FrozenSet['cirq.MeasurementKey']:
    """Gets the keys that the value is classically controlled by.

    Args:
        val: The object that may be classically controlled.

    Returns:
        The measurement keys the value is controlled by. If the value is not
        classically controlled, the result is the empty tuple.
    """
    getter = getattr(val, '_control_keys_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result is not None:
        return result
    return frozenset()