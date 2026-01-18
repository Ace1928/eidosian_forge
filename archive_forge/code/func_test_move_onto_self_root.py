import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_onto_self_root(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    tree.add(['a'])
    tree.commit('initial')
    self.assertRaises(errors.BzrMoveFailedError, tree.move, ['a'], 'a')
    tree._validate()