import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_dir_export(self):
    tree = self.make_branch_and_tree('dir')
    self.build_tree(['dir/a'])
    tree.add('a')
    self.build_tree_contents([('dir/.bzrrules', b'')])
    tree.add('.bzrrules')
    self.build_tree(['dir/.bzr-adir/', 'dir/.bzr-adir/afile'])
    tree.add(['.bzr-adir/', '.bzr-adir/afile'])
    os.chdir('dir')
    self.run_bzr('ignore something')
    tree.commit('1')
    self.assertTrue(tree.has_filename('.bzrignore'))
    self.assertTrue(tree.has_filename('.bzrrules'))
    self.assertTrue(tree.has_filename('.bzr-adir'))
    self.assertTrue(tree.has_filename('.bzr-adir/afile'))
    self.run_bzr('export direxport')
    files = sorted(os.listdir('direxport'))
    self.assertEqual(['a'], files)