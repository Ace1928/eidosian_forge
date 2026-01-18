import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def dict_insert(self, key_bytes, val_bytes):
    hashval = hash(key_bytes)
    status = self.tc.numba_dict_insert_ez(self.dp, key_bytes, hashval, val_bytes)
    self.tc.assertGreaterEqual(status, 0)