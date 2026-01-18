from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def is_measurement(val: Any) -> bool:
    """Determines whether or not the given value is a measurement (or contains one).

    Measurements are identified by the fact that any of them may have an `_is_measurement_` method
    or `cirq.measurement_keys` returns a non-empty result for them.

    Args:
        val: The value which to evaluate.
        allow_decompose: Defaults to True. When true, composite operations that
            don't directly specify their `_is_measurement_` property will be decomposed in
            order to find any measurements keys within the decomposed operations.
    """
    result = _is_measurement_from_magic_method(val)
    if isinstance(result, bool):
        return result
    keys = measurement_key_objs(val)
    return keys is not NotImplemented and bool(keys)