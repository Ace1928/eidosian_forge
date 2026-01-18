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
def test_cython_inline(self):
    ip = self._ip
    ip.ex('a=10; b=20')
    result = ip.run_cell_magic('cython_inline', '', 'return a+b')
    self.assertEqual(result, 30)