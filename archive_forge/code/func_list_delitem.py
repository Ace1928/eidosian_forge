import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def list_delitem(self, i):
    if isinstance(i, slice):
        status = self.tc.numba_list_delete_slice(self.lp, i.start, i.stop, i.step)
        if status == LIST_ERR_IMMUTABLE:
            raise ValueError('list is immutable')
        self.tc.assertEqual(status, LIST_OK)
    else:
        i = self.handle_index(i)
        status = self.tc.numba_list_delitem(self.lp, i)
        if status == LIST_ERR_INDEX:
            raise IndexError('list index out of range')
        elif status == LIST_ERR_IMMUTABLE:
            raise ValueError('list is immutable')
        self.tc.assertEqual(status, LIST_OK)