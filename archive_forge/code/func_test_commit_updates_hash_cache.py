import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_commit_updates_hash_cache(self):
    tree = self.get_tree_with_cachable_file_foo()
    tree.commit('a commit')
    entry = tree._get_entry(path='foo')
    expected_sha1 = osutils.sha_file_by_name('foo')
    self.assertEqual(expected_sha1, entry[1][0][1])
    self.assertEqual(len('a bit of content for foo\n'), entry[1][0][2])