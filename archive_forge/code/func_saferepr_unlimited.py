import pprint
import reprlib
from typing import Optional
def saferepr_unlimited(obj: object, use_ascii: bool=True) -> str:
    """Return an unlimited-size safe repr-string for the given object.

    As with saferepr, failing __repr__ functions of user instances
    will be represented with a short exception info.

    This function is a wrapper around simple repr.

    Note: a cleaner solution would be to alter ``saferepr``this way
    when maxsize=None, but that might affect some other code.
    """
    try:
        if use_ascii:
            return ascii(obj)
        return repr(obj)
    except Exception as exc:
        return _format_repr_exception(exc, obj)