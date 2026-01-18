import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_dir_export_nested(self):
    tree = self.make_branch_and_tree('dir')
    self.build_tree(['dir/a'])
    tree.add('a')
    subtree = self.make_branch_and_tree('dir/subdir')
    tree.add_reference(subtree)
    self.build_tree(['dir/subdir/b'])
    subtree.add('b')
    self.run_bzr('export --uncommitted direxport1 dir')
    self.assertFalse(os.path.exists('direxport1/subdir/b'))
    self.run_bzr('export --recurse-nested --uncommitted direxport2 dir')
    self.assertTrue(os.path.exists('direxport2/subdir/b'))