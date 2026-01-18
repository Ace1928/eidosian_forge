import os
import shutil
import stat
from dulwich.objects import Blob, Tree
from ...branchbuilder import BranchBuilder
from ...bzr.inventory import InventoryDirectory, InventoryFile
from ...errors import NoSuchRevision
from ...graph import DictParentsProvider, Graph
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import SymlinkFeature
from ..cache import DictGitShaMap
from ..object_store import (BazaarObjectStore, LRUTreeCache,
def test_directory_converted_to_symlink(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    b = Blob()
    b.data = b'trgt'
    self.store.lock_read()
    self.addCleanup(self.store.unlock)
    self.assertRaises(KeyError, self.store.__getitem__, b.id)
    tree = self.branch.controldir.create_workingtree()
    self.build_tree_contents([('foo/',), ('foo/bar', b'a\nb\nc\nd\ne\n')])
    tree.add(['foo', 'foo/bar'])
    revid1 = tree.commit('commit 1')
    shutil.rmtree('foo')
    os.symlink('trgt', 'foo')
    revid2 = tree.commit('commit 2')
    self.assertRaises(KeyError, self.store.__getitem__, b.id)
    self.store.unlock()
    self.store.lock_read()
    self.assertEqual(b, self.store[b.id])