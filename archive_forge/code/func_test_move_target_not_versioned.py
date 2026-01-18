import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_target_not_versioned(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'b'])
    tree.add(['b'])
    tree.commit('initial')
    if tree.has_versioned_directories():
        self.assertRaises(errors.BzrMoveFailedError, tree.move, ['b'], 'a')
    else:
        tree.move(['b'], 'a')
    tree._validate()