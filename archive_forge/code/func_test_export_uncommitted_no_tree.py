import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_export_uncommitted_no_tree(self):
    """Test --uncommitted option only works with a working tree."""
    tree = self.example_branch()
    tree.controldir.destroy_workingtree()
    os.chdir('branch')
    self.run_bzr_error(['brz: ERROR: --uncommitted requires a working tree'], 'export --uncommitted latest')