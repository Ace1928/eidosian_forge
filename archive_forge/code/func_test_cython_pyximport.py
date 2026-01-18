from __future__ import absolute_import
import os
import io
import sys
from contextlib import contextmanager
from unittest import skipIf
from Cython.Build import IpythonMagic
from Cython.TestUtils import CythonTest
from Cython.Compiler.Annotate import AnnotationCCodeWriter
from libc.math cimport sin
@skip_win32
def test_cython_pyximport(self):
    ip = self._ip
    module_name = '_test_cython_pyximport'
    ip.run_cell_magic('cython_pyximport', module_name, code)
    ip.ex('g = f(10)')
    self.assertEqual(ip.user_ns['g'], 20.0)
    ip.run_cell_magic('cython_pyximport', module_name, code)
    ip.ex('h = f(-10)')
    self.assertEqual(ip.user_ns['h'], -20.0)
    try:
        os.remove(module_name + '.pyx')
    except OSError:
        pass