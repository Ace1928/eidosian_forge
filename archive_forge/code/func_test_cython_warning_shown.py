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
def test_cython_warning_shown(self):
    ip = self._ip
    with capture_output() as out:
        ip.run_cell_magic('cython', '-3 -f', compile_warning_code)
    captured_out, captured_err = out
    self.assertTrue('CWarning' in captured_out)