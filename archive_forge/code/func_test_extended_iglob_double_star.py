import contextlib
import os.path
import sys
import tempfile
import unittest
from io import open
from os.path import join as pjoin
from ..Dependencies import extended_iglob
def test_extended_iglob_double_star(self):
    basedirs = os.listdir('.')
    files = [pjoin(basedir, dirname, filename) for basedir in basedirs for dirname in 'xyz' for filename in ['file2_pyx.pyx', 'file2_py.py']]
    all_files = [pjoin(basedir, filename) for basedir in basedirs for filename in ['file1_pyx.pyx', 'file1_py.py']] + files
    self.files_equal('*/*/*', files)
    self.files_equal('*/*/**/*', files)
    self.files_equal('*/**/*.*', all_files)
    self.files_equal('**/*.*', all_files)
    self.files_equal('*/**/*.c12', [])
    self.files_equal('**/*.c12', [])
    self.files_equal('*/*/*.{py,pyx,c12}', files)
    self.files_equal('*/*/**/*.{py,pyx,c12}', files)
    self.files_equal('*/**/*/*.{py,pyx,c12}', files)
    self.files_equal('**/*/*/*.{py,pyx,c12}', files)
    self.files_equal('**/*.{py,pyx,c12}', all_files)
    self.files_equal('*/*/*.{py,pyx}', files)
    self.files_equal('**/*/*/*.{py,pyx}', files)
    self.files_equal('*/**/*/*.{py,pyx}', files)
    self.files_equal('**/*.{py,pyx}', all_files)
    self.files_equal('*/*/*.{pyx}', files[::2])
    self.files_equal('**/*.{pyx}', all_files[::2])
    self.files_equal('*/**/*/*.pyx', files[::2])
    self.files_equal('*/*/*.pyx', files[::2])
    self.files_equal('**/*.pyx', all_files[::2])
    self.files_equal('*/*/*.{py}', files[1::2])
    self.files_equal('**/*.{py}', all_files[1::2])
    self.files_equal('*/*/*.py', files[1::2])
    self.files_equal('**/*.py', all_files[1::2])