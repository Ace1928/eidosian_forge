import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_dir_export_per_file_timestamps(self):
    tree = self.example_branch()
    self.build_tree_contents([('branch/har', b'foo')])
    tree.add('har')
    tree.commit('setup', timestamp=315532800)
    self.run_bzr('export --per-file-timestamps t branch')
    har_st = os.stat('t/har')
    self.assertEqual(315532800, har_st.st_mtime)