import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def make_test_tree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('link@', 'tree'), ('tree/outerlink@', '/not/there'), ('tree/content', b'hello'), ('tree/sublink@', 'subdir'), ('tree/subdir/',), ('tree/subdir/subcontent', b'subcontent stuff')])