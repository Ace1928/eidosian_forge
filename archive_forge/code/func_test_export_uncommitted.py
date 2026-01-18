import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_export_uncommitted(self):
    """Test --uncommitted option"""
    self.example_branch()
    os.chdir('branch')
    self.build_tree_contents([('goodbye', b'uncommitted data')])
    self.run_bzr(['export', '--uncommitted', 'latest'])
    self.check_file_contents('latest/goodbye', b'uncommitted data')