import os
from breezy import errors, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.tree import FileTimestampUnavailable
def test_get_file_mtime_nested(self):
    tree = self.make_basic_tree()
    subtree = self.make_branch_and_tree('tree/sub')
    self.build_tree(['tree/sub/one'])
    subtree.add(['one'])
    subtree.commit('one')
    try:
        tree.add_reference(subtree)
    except errors.UnsupportedOperation:
        raise TestNotApplicable('subtrees not supported')
    tree.commit('sub')
    with tree.lock_read(), subtree.lock_read():
        self.assertEqual(tree.get_file_mtime('sub/one'), subtree.get_file_mtime('one'))
        self.assertEqual(tree.basis_tree().get_file_mtime('sub/one'), subtree.basis_tree().get_file_mtime('one'))