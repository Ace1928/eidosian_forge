import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def dict_new_minsize(self, key_size, val_size):
    dp = ctypes.c_void_p()
    status = self.tc.numba_dict_new_sized(ctypes.byref(dp), 0, key_size, val_size)
    self.tc.assertEqual(status, 0)
    return dp