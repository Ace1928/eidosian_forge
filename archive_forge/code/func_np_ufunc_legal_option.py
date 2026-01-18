import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def np_ufunc_legal_option(key, value):
    """Checking if ufunc arguments are legal inputs

    Parameters
    ----------
    key : string
        the key of the ufunc argument.
    value : string
        the value of the ufunc argument.

    Returns
    -------
    legal : boolean
        Whether or not the argument is a legal one. True when the key is one of the ufunc
        arguments and value is an allowed value. False when the key is not one of the ufunc
        arugments or the value is not an allowed value even when the key is a legal one.
    """
    if key == 'where':
        return True
    elif key == 'casting':
        return value in set(['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
    elif key == 'order':
        if isinstance(value, str):
            return True
    elif key == 'dtype':
        import numpy as _np
        return value in set([_np.int8, _np.uint8, _np.int32, _np.int64, _np.float16, _np.float32, _np.float64, 'int8', 'uint8', 'int32', 'int64', 'float16', 'float32', 'float64'])
    elif key == 'subok':
        return isinstance(value, bool)
    return False