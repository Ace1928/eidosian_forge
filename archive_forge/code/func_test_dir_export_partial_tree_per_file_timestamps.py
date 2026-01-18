import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_dir_export_partial_tree_per_file_timestamps(self):
    tree = self.example_branch()
    self.build_tree(['branch/subdir/', 'branch/subdir/foo.txt'])
    tree.smart_add(['branch'])
    tree.commit('setup', timestamp=315532800)
    self.run_bzr('export --per-file-timestamps tpart branch/subdir')
    foo_st = os.stat('tpart/foo.txt')
    self.assertEqual(315532800, foo_st.st_mtime)