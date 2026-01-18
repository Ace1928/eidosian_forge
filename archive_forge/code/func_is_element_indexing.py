from collections import namedtuple
import itertools
import functools
import operator
import ctypes
import numpy as np
from numba import _helperlib
from numba.core import config
def is_element_indexing(item, ndim):
    if isinstance(item, slice):
        return False
    elif isinstance(item, tuple):
        if len(item) == ndim:
            if not any((isinstance(it, slice) for it in item)):
                return True
    else:
        return True
    return False