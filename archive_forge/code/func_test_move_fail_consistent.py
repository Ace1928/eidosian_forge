import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_fail_consistent(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/', 'b/a', 'c'])
    tree.add(['a', 'b', 'c'])
    tree.commit('initial')
    self.assertRaises(errors.RenameFailedFilesExist, tree.move, ['c', 'a'], 'b')
    if osutils.lexists('c'):
        self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/'), ('c', 'c')])
    else:
        self.assertPathExists('b/c')
        self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/'), ('b/c', 'c')])
    tree._validate()