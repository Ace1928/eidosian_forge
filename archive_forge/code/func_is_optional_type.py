import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def is_optional_type(tp):
    """Test if the type is type(None), or is a direct union with it, such as Optional[T].

    NOTE: this method inspects nested `Union` arguments but not `TypeVar` definition
    bounds and constraints. So it will return `False` if
     - `tp` is a `TypeVar` bound, or constrained to, an optional type
     - `tp` is a `Union` to a `TypeVar` bound or constrained to an optional type,
     - `tp` refers to a *nested* `Union` containing an optional type or one of the above.

    Users wishing to check for optionality in types relying on type variables might wish
    to use this method in combination with `get_constraints` and `get_bound`
    """
    if tp is type(None):
        return True
    elif is_union_type(tp):
        return any((is_optional_type(tt) for tt in get_args(tp, evaluate=True)))
    else:
        return False