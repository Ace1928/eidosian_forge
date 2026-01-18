import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def make_tree_with_extra_bzr_files(self):
    tree = self.make_basic_tree()
    self.build_tree_contents([('tree/.bzrrules', b'')])
    self.build_tree(['tree/.bzr-adir/', 'tree/.bzr-adir/afile'])
    tree.add(['.bzrrules', '.bzr-adir/', '.bzr-adir/afile'])
    self.run_bzr('ignore something -d tree')
    tree.commit('2')
    return tree