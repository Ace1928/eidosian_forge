from typing import Any
def is_tripwire(obj: Any) -> bool:
    """Returns True if `obj` appears to be a TripWire object

    Examples
    --------
    >>> is_tripwire(object())
    False
    >>> is_tripwire(TripWire('some message'))
    True
    """
    try:
        obj.any_attribute
    except TripWireError:
        return True
    except Exception:
        pass
    return False