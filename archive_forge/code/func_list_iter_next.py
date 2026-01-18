import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def list_iter_next(self, itptr):
    bi = ctypes.c_void_p(0)
    status = self.tc.numba_list_iter_next(itptr, ctypes.byref(bi))
    if status == LIST_ERR_MUTATED:
        raise ValueError('list mutated')
    elif status == LIST_ERR_ITER_EXHAUSTED:
        raise StopIteration
    else:
        self.tc.assertGreaterEqual(status, 0)
        item = (ctypes.c_char * self.item_size).from_address(bi.value)
        return item.value