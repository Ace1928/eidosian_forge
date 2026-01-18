import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def npy_ctypes_check(cls):
    try:
        if IS_PYPY:
            ctype_base = cls.__mro__[-3]
        else:
            ctype_base = cls.__mro__[-2]
        return '_ctypes' in ctype_base.__module__
    except Exception:
        return False