import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def test_insertion_many(self):
    self.check_insertion_many(nmax=7)
    self.check_insertion_many(nmax=8)
    self.check_insertion_many(nmax=9)
    self.check_insertion_many(nmax=31)
    self.check_insertion_many(nmax=32)
    self.check_insertion_many(nmax=33)
    self.check_insertion_many(nmax=1023)
    self.check_insertion_many(nmax=1024)
    self.check_insertion_many(nmax=1025)
    self.check_insertion_many(nmax=4095)
    self.check_insertion_many(nmax=4096)
    self.check_insertion_many(nmax=4097)