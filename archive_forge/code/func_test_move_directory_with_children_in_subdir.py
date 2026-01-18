import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_directory_with_children_in_subdir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'a/c/', 'd/'])
    tree.add(['a', 'a/b', 'a/c', 'd'])
    tree.commit('initial')
    tree.rename_one('a/b', 'a/c/b')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a/', 'a/'), ('d/', 'd/'), ('a/c/', 'a/c/'), ('a/c/b', 'a/b')])
    self.assertEqual([('a', 'd/a')], tree.move(['a'], 'd'))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('d/', 'd/'), ('d/a/', 'a/'), ('d/a/c/', 'a/c/'), ('d/a/c/b', 'a/b')])
    tree._validate()