import cmath
import numpy as np
from numba import float32
from numba.types import unicode_type, i8
from numba.pycc import CC, exportmany, export
from numba.tests.support import has_blas
from numba import typed
@cc_helperlib.export('np_sqrt', 'f8(f8)')
def np_sqrt(u):
    return np.sqrt(u)