import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
def check_stencil_arrays(self, *args, **kwargs):
    neighborhood = kwargs.get('neighborhood')
    init_shape = args[0].shape
    if neighborhood is not None:
        if len(init_shape) != len(neighborhood):
            raise ValueError('Invalid neighborhood supplied')
    for x in args[1:]:
        if hasattr(x, 'shape'):
            if init_shape != x.shape:
                raise ValueError('Input stencil arrays do not commute')