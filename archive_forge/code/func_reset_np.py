import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def reset_np():
    """Deactivate NumPy shape and array semantics at the same time."""
    set_np(shape=False, array=False)