import unittest, os, errno
import threading
from ctypes import *
from ctypes.util import find_library
@unittest.skipUnless(os.name == 'nt', 'Test specific to Windows')
def test_GetLastError(self):
    dll = WinDLL('kernel32', use_last_error=True)
    GetModuleHandle = dll.GetModuleHandleA
    GetModuleHandle.argtypes = [c_wchar_p]
    self.assertEqual(0, GetModuleHandle('foo'))
    self.assertEqual(get_last_error(), 126)
    self.assertEqual(set_last_error(32), 126)
    self.assertEqual(get_last_error(), 32)

    def _worker():
        set_last_error(0)
        dll = WinDLL('kernel32', use_last_error=False)
        GetModuleHandle = dll.GetModuleHandleW
        GetModuleHandle.argtypes = [c_wchar_p]
        GetModuleHandle('bar')
        self.assertEqual(get_last_error(), 0)
    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    self.assertEqual(get_last_error(), 32)
    set_last_error(0)