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
def test_nested_tree(self):
    wt = self.make_branch_and_tree('.', format='development-subtree')
    subtree = self.make_branch_and_tree('subtree')
    self.build_tree(['subtree/file'])
    subtree.add(['file'])
    wt.add(['subtree'])
    export.export(wt, 'target', format='dir')
    self.assertPathExists('target/subtree')