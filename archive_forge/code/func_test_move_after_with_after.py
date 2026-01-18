import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_after_with_after(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/'])
    tree.add(['a', 'b'])
    tree.commit('initial')
    os.rename('a', 'b/a')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/')])
    self.assertEqual([('a', 'b/a')], tree.move(['a'], 'b', after=True))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/'), ('b/a', 'a')])
    tree._validate()