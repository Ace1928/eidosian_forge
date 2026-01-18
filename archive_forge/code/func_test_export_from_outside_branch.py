import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_export_from_outside_branch(self):
    self.example_branch()
    self.run_bzr('export latest branch')
    self.assertEqual(['goodbye', 'hello'], sorted(os.listdir('latest')))
    self.check_file_contents('latest/goodbye', b'baz')
    self.run_bzr('export first -r 1 branch')
    self.assertEqual(['hello'], sorted(os.listdir('first')))
    self.check_file_contents('first/hello', b'foo')