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
def test_cython_annotate_complete_c_code(self):
    ip = self._ip
    html = ip.run_cell_magic('cython', '--annotate-fullc', code)
    self.assertTrue(AnnotationCCodeWriter.COMPLETE_CODE_TITLE in html.data)