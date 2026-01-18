import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def is_np_shape():
    """Checks whether the NumPy shape semantics is currently turned on.
    In NumPy shape semantics, `()` represents the shape of scalar tensors,
    and tuples with `0` elements, for example, `(0,)`, `(1, 0, 2)`, represent
    the shapes of zero-size tensors. This is turned off by default for keeping
    backward compatibility.

    In the NumPy shape semantics, `-1` indicates an unknown size. For example,
    `(-1, 2, 2)` means that the size of the first dimension is unknown. Its size
    may be inferred during shape inference.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy within this semantics.

    Returns
    -------
        A bool value indicating whether the NumPy shape semantics is currently on.

    Example
    -------
    >>> import mxnet as mx
    >>> prev_state = mx.set_np_shape(True)
    >>> print(prev_state)
    False
    >>> print(mx.is_np_shape())
    True
    """
    curr = ctypes.c_bool()
    check_call(_LIB.MXIsNumpyShape(ctypes.byref(curr)))
    return curr.value