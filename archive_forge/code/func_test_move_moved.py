import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_moved(self):
    """Moving a moved entry works as expected."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'c/'])
    tree.add(['a', 'a/b', 'c'])
    tree.commit('initial')
    self.assertEqual([('a/b', 'c/b')], tree.move(['a/b'], 'c'))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a/', 'a/'), ('c/', 'c/'), ('c/b', 'a/b')])
    self.assertEqual([('c/b', 'b')], tree.move(['c/b'], ''))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a/', 'a/'), ('b', 'a/b'), ('c/', 'c/')])
    tree._validate()