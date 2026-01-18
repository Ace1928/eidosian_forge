import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_basic_zipfile_export(self):
    self.example_branch()
    os.chdir('branch')
    self.run_bzr('export ../first.zip -r 1')
    self.assertPathExists('../first.zip')
    with zipfile.ZipFile('../first.zip') as zf:
        self.assertEqual(['first/hello'], sorted(zf.namelist()))
        self.assertEqual(b'foo', zf.read('first/hello'))
    self.run_bzr('export ../first2.zip -r 1 --root pizza')
    with zipfile.ZipFile('../first2.zip') as zf:
        self.assertEqual(['pizza/hello'], sorted(zf.namelist()))
        self.assertEqual(b'foo', zf.read('pizza/hello'))
    self.run_bzr('export ../first-zip --format=zip -r 1')
    with zipfile.ZipFile('../first-zip') as zf:
        self.assertEqual(['first-zip/hello'], sorted(zf.namelist()))
        self.assertEqual(b'foo', zf.read('first-zip/hello'))