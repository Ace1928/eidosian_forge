import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def check_insertion_many(self, nmax):
    d = Dict(self, 8, 8)

    def make_key(v):
        return 'key_{:04}'.format(v)

    def make_val(v):
        return 'val_{:04}'.format(v)
    for i in range(nmax):
        d[make_key(i)] = make_val(i)
        self.assertEqual(len(d), i + 1)
    for i in range(nmax):
        self.assertEqual(d[make_key(i)], make_val(i))