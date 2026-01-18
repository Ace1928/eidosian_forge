import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_conversion_setitem(self, flags=enable_pyobj_flags):
    """ this used to work, and was used in one of the tutorials """
    from numba import jit

    def pyfunc(array):
        for index in range(len(array)):
            array[index] = index % decimal.Decimal(100)
    cfunc = jit('void(i8[:])', **flags)(pyfunc)
    udt = np.arange(100, dtype='i1')
    control = udt.copy()
    pyfunc(control)
    cfunc(udt)
    self.assertPreciseEqual(udt, control)