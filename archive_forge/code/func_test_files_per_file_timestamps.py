import gzip
import os
import tarfile
import time
import zipfile
from io import BytesIO
from .. import errors, export, tests
from ..archive.tar import tarball_generator
from ..export import get_root_name
from . import features
def test_files_per_file_timestamps(self):
    builder = self.make_branch_builder('source')
    builder.start_series()
    a_time = time.mktime((1999, 12, 12, 0, 0, 0, 0, 0, 0))
    b_time = time.mktime((1980, 1, 1, 0, 0, 0, 0, 0, 0))
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('a', b'a-id', 'file', b'content\n'))], timestamp=a_time)
    builder.build_snapshot(None, [('add', ('b', b'b-id', 'file', b'content\n'))], timestamp=b_time)
    builder.finish_series()
    b = builder.get_branch()
    b.lock_read()
    self.addCleanup(b.unlock)
    tree = b.basis_tree()
    export.export(tree, 'target', format='dir', per_file_timestamps=True)
    t = self.get_transport('target')
    self.assertEqual(a_time, t.stat('a').st_mtime)
    self.assertEqual(b_time, t.stat('b').st_mtime)