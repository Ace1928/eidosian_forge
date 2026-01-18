from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
def exch(self, array, index, val):
    with exchlock:
        old = array[index]
        array[index] = val
    return old