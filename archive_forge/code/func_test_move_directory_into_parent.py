import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_directory_into_parent(self):
    if not self.workingtree_format.supports_versioned_directories:
        raise tests.TestNotApplicable('test requires versioned directories')
    tree = self.make_branch_and_tree('.')
    self.build_tree(['c/', 'c/b/', 'c/b/d/'])
    tree.add(['c', 'c/b', 'c/b/d'])
    tree.commit('initial')
    self.assertEqual([('c/b', 'b')], tree.move(['c/b'], ''))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'c/b/'), ('c/', 'c/'), ('b/d/', 'c/b/d/')])
    tree._validate()