import contextlib
import os.path
import sys
import tempfile
import unittest
from io import open
from os.path import join as pjoin
from ..Dependencies import extended_iglob
def test_extended_iglob_simple(self):
    ax_files = [pjoin('a', 'x', 'file2_pyx.pyx'), pjoin('a', 'x', 'file2_py.py')]
    self.files_equal('a/x/*', ax_files)
    self.files_equal('a/x/*.c12', [])
    self.files_equal('a/x/*.{py,pyx,c12}', ax_files)
    self.files_equal('a/x/*.{py,pyx}', ax_files)
    self.files_equal('a/x/*.{pyx}', ax_files[:1])
    self.files_equal('a/x/*.pyx', ax_files[:1])
    self.files_equal('a/x/*.{py}', ax_files[1:])
    self.files_equal('a/x/*.py', ax_files[1:])