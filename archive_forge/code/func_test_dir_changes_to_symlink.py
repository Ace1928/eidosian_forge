import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_dir_changes_to_symlink(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/a/',), ('tree/a/file', b'content')])
    tree.smart_add(['tree/a'])
    tree.commit('add dir')
    osutils.rmtree('tree/a')
    self.build_tree_contents([('tree/a@', 'target')])
    tree.commit('change to symlink')