import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_through_symlinks(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b'])
    os.symlink('a', 'c')
    os.symlink('.', 'd')
    tree.add(['a', 'a/b', 'c'], ids=[b'a-id', b'b-id', b'c-id'])
    self.run_bzr('mv c/b b')
    tree = workingtree.WorkingTree.open('.')
    self.assertEqual(b'b-id', tree.path2id('b'))