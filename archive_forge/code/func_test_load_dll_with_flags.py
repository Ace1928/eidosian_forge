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
@unittest.skipUnless(os.name == 'nt', 'test specific to Windows')
def test_load_dll_with_flags(self):
    _sqlite3 = import_helper.import_module('_sqlite3')
    src = _sqlite3.__file__
    if src.lower().endswith('_d.pyd'):
        ext = '_d.dll'
    else:
        ext = '.dll'
    with os_helper.temp_dir() as tmp:
        target = os.path.join(tmp, '_sqlite3.dll')
        shutil.copy(src, target)
        shutil.copy(os.path.join(os.path.dirname(src), 'sqlite3' + ext), os.path.join(tmp, 'sqlite3' + ext))

        def should_pass(command):
            with self.subTest(command):
                subprocess.check_output([sys.executable, '-c', 'from ctypes import *; import nt;' + command], cwd=tmp)

        def should_fail(command):
            with self.subTest(command):
                with self.assertRaises(subprocess.CalledProcessError):
                    subprocess.check_output([sys.executable, '-c', 'from ctypes import *; import nt;' + command], cwd=tmp, stderr=subprocess.STDOUT)
        should_fail("WinDLL('_sqlite3.dll')")
        should_pass("WinDLL('./_sqlite3.dll')")
        should_pass("windll.kernel32.SetDllDirectoryW(None); WinDLL('_sqlite3.dll', winmode=0)")
        should_fail("WinDLL(nt._getfullpathname('_sqlite3.dll'), " + 'winmode=nt._LOAD_LIBRARY_SEARCH_SYSTEM32)')
        should_pass("WinDLL(nt._getfullpathname('_sqlite3.dll'), " + 'winmode=nt._LOAD_LIBRARY_SEARCH_SYSTEM32|' + 'nt._LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)')
        should_pass('import os; p = os.add_dll_directory(os.getcwd());' + "WinDLL('_sqlite3.dll'); p.close()")