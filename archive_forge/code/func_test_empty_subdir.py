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
def test_empty_subdir(self):
    self.build_tree(['source/', 'source/a', 'source/b/', 'source/b/c'])
    wt = self.make_branch_and_tree('source')
    wt.add(['a', 'b', 'b/c'])
    wt.commit('1')
    self.build_tree(['target/'])
    export.export(wt, 'target', format='dir', subdir='')
    self.assertPathExists('target/a')
    self.assertPathExists('target/b')
    self.assertPathExists('target/b/c')