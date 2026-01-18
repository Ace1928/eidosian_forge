import ctypes
import logging
import os
import platform
import shutil
import stat
import sys
import tempfile
import subprocess
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.common.envvar as envvar
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import (
from pyomo.common.download import FileDownloader
def test_find_library_system(self):
    _args = {'cwd': False, 'include_PATH': False, 'pathlist': []}
    if FileDownloader.get_sysinfo()[0] == 'windows':
        a = find_library('ntdll', **_args)
        b = find_library('ntdll.dll', **_args)
        c = find_library('foo\\bar\\ntdll.dll', **_args)
    else:
        a = find_library('c', **_args)
        b = find_library('libc.so', **_args)
        c = find_library('foo/bar/libc.so', **_args)
    self.assertIsNotNone(a)
    self.assertIsNotNone(b)
    self.assertIsNotNone(c)
    self.assertEqual(a, b)
    self.assertTrue(c.startswith(a))
    _lib = ctypes.cdll.LoadLibrary(a)
    self.assertIsNotNone(_lib)