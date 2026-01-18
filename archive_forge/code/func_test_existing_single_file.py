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
def test_existing_single_file(self):
    self.build_tree(['dir1/', 'dir1/dir2/', 'dir1/first', 'dir1/dir2/second'])
    wtree = self.make_branch_and_tree('dir1')
    wtree.add(['dir2', 'first', 'dir2/second'])
    wtree.commit('1')
    export.export(wtree, 'target1', format='dir', subdir='first')
    self.assertPathExists('target1/first')
    export.export(wtree, 'target2', format='dir', subdir='dir2/second')
    self.assertPathExists('target2/second')