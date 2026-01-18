from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def pvector_field(item_type, optional=False, initial=(), invariant=PFIELD_NO_INVARIANT, item_invariant=PFIELD_NO_INVARIANT):
    """
    Create checked ``PVector`` field.

    :param item_type: The required type for the items in the vector.
    :param optional: If true, ``None`` can be used as a value for
        this field.
    :param initial: Initial value to pass to factory if no value is given
        for the field.

    :return: A ``field`` containing a ``CheckedPVector`` of the given type.
    """
    return _sequence_field(CheckedPVector, item_type, optional, initial, invariant=invariant, item_invariant=item_invariant)