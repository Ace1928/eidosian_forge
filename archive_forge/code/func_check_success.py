import os
from breezy import osutils, tests, workingtree
def check_success(self, path):
    base_tree = workingtree.WorkingTree.open(path)
    self.assertEqual(b'file1-id', base_tree.path2id('subtree/file1'))