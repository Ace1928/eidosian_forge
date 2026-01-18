from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
def gridsize(self, n):
    bdim = self.blockDim
    gdim = self.gridDim
    x = bdim.x * gdim.x
    if n == 1:
        return x
    y = bdim.y * gdim.y
    if n == 2:
        return (x, y)
    z = bdim.z * gdim.z
    if n == 3:
        return (x, y, z)
    raise RuntimeError('Global grid has 1-3 dimensions. %d requested' % n)