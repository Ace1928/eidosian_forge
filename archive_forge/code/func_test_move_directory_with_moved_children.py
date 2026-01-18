import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_directory_with_moved_children(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'a/c', 'd', 'e/'])
    tree.add(['a', 'a/b', 'a/c', 'd', 'e'])
    tree.commit('initial')
    self.assertEqual([('a/b', 'b')], tree.move(['a/b'], ''))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a/', 'a/'), ('b', 'a/b'), ('d', 'd'), ('e/', 'e/'), ('a/c', 'a/c')])
    self.assertEqual([('d', 'a/d')], tree.move(['d'], 'a'))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a/', 'a/'), ('b', 'a/b'), ('e/', 'e/'), ('a/c', 'a/c'), ('a/d', 'd')])
    self.assertEqual([('a', 'e/a')], tree.move(['a'], 'e'))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b', 'a/b'), ('e/', 'e/'), ('e/a/', 'a/'), ('e/a/c', 'a/c'), ('e/a/d', 'd')])
    tree._validate()