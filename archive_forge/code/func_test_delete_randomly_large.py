import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def test_delete_randomly_large(self):
    self.check_delete_randomly(nmax=2 ** 17, ndrop=2 ** 16, nrefill=2 ** 10)