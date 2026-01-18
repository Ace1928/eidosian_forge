import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def dict_popitem(self):
    key_bytes = ctypes.create_string_buffer(self.keysize)
    val_bytes = ctypes.create_string_buffer(self.valsize)
    status = self.tc.numba_dict_popitem(self.dp, key_bytes, val_bytes)
    if status != 0:
        if status == -4:
            raise KeyError('popitem(): dictionary is empty')
        else:
            self.tc._fail('Unknown')
    return (key_bytes.value, val_bytes.value)