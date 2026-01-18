import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_invalid_rename(self):
    tree = self.create_wt4()
    with tree.lock_write():
        tree.commit('init')
        state = tree.current_dirstate()
        state._read_dirblocks_if_needed()
        state._dirblocks[1][1].append(((b'', b'foo', b'foo-id'), [(b'f', b'', 0, False, b''), (b'r', b'bar', 0, False, b'')]))
        self.assertListRaises(dirstate.DirstateCorrupt, tree.iter_changes, tree.basis_tree())