import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_subdir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/', 'b/c'])
    tree.add(['a', 'b', 'b/c'])
    tree.commit('initial')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/'), ('b/c', 'b/c')])
    a_contents = tree.get_file_text('a')
    self.assertEqual([('a', 'b/a')], tree.move(['a'], 'b'))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/'), ('b/a', 'a'), ('b/c', 'b/c')])
    self.assertPathDoesNotExist('a')
    self.assertFileEqual(a_contents, 'b/a')
    tree._validate()