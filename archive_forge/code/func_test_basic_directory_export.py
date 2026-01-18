import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_basic_directory_export(self):
    self.example_branch()
    os.chdir('branch')
    self.run_bzr('export ../latest')
    self.assertEqual(['goodbye', 'hello'], sorted(os.listdir('../latest')))
    self.check_file_contents('../latest/goodbye', b'baz')
    self.run_bzr('export ../first -r 1')
    self.assertEqual(['hello'], sorted(os.listdir('../first')))
    self.check_file_contents('../first/hello', b'foo')
    self.run_bzr('export ../first.gz -r 1')
    self.check_file_contents('../first.gz/hello', b'foo')
    self.run_bzr('export ../first.bz2 -r 1')
    self.check_file_contents('../first.bz2/hello', b'foo')