from ctypes import *
import os
import shutil
import subprocess
import sys
import unittest
import test.support
from test.support import import_helper
from test.support import os_helper
from ctypes.util import find_library
@unittest.skipUnless(os.name == 'nt', 'Windows-specific test')
def test_1703286_B(self):
    from _ctypes import call_function
    advapi32 = windll.advapi32
    self.assertEqual(0, advapi32.CloseEventLog(None))
    windll.kernel32.GetProcAddress.argtypes = (c_void_p, c_char_p)
    windll.kernel32.GetProcAddress.restype = c_void_p
    proc = windll.kernel32.GetProcAddress(advapi32._handle, b'CloseEventLog')
    self.assertTrue(proc)
    self.assertEqual(0, call_function(proc, (None,)))