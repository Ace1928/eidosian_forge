import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_export_directory(self):
    """Test --directory option"""
    self.example_branch()
    self.run_bzr(['export', '--directory=branch', 'latest'])
    self.assertEqual(['goodbye', 'hello'], sorted(os.listdir('latest')))
    self.check_file_contents('latest/goodbye', b'baz')