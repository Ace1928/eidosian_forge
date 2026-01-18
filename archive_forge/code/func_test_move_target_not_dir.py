import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_target_not_dir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    tree.add(['a'])
    tree.commit('initial')
    self.assertRaises(errors.BzrMoveFailedError, tree.move, ['a'], 'not-a-dir')
    tree._validate()