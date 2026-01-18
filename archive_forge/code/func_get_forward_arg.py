import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def get_forward_arg(fr):
    """
    If fr is a ForwardRef, return the string representation of the forward reference.
    Otherwise return None. Examples::

        tp = List["FRef"]
        fr = get_args(tp)[0]
        get_forward_arg(fr) == "FRef"
        get_forward_arg(tp) == None
    """
    return fr.__forward_arg__ if is_forward_ref(fr) else None