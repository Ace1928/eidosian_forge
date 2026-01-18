import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def test_delete_randomly(self):
    self.check_delete_randomly(nmax=8, ndrop=2, nrefill=2)
    self.check_delete_randomly(nmax=13, ndrop=10, nrefill=31)
    self.check_delete_randomly(nmax=100, ndrop=50, nrefill=200)
    self.check_delete_randomly(nmax=100, ndrop=99, nrefill=100)
    self.check_delete_randomly(nmax=100, ndrop=100, nrefill=100)
    self.check_delete_randomly(nmax=1024, ndrop=999, nrefill=1)
    self.check_delete_randomly(nmax=1024, ndrop=999, nrefill=2048)