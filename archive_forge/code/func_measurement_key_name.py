from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def measurement_key_name(val: Any, default: Any=RaiseTypeErrorIfNotProvided):
    """Get the single measurement key for the given value.

    Args:
        val: The value which has one measurement key.
        default: Determines the fallback behavior when `val` doesn't have
            a measurement key. If `default` is not set, a TypeError is raised.
            If default is set to a value, that value is returned if the value
            does not have `_measurement_key_name_`.

    Returns:
        If `val` has a `_measurement_key_name_` method and its result is not
        `NotImplemented`, that result is returned. Otherwise, if a default
        value was specified, the default value is returned.

    Raises:
        TypeError: `val` doesn't have a _measurement_key_name_ method (or that method
            returned NotImplemented) and also no default value was specified.
        ValueError: `val` has multiple measurement keys.
    """
    result = measurement_key_names(val)
    if len(result) == 1:
        return next(iter(result))
    if len(result) > 1:
        raise ValueError(f'Got multiple measurement keys ({result!r}) from {val!r}.')
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(f"Object of type '{type(val)}' had no measurement keys.")