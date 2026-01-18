import os
from breezy import osutils, tests, workingtree
def make_trees(self):
    base_tree = self.make_branch_and_tree('tree', format='development-subtree')
    base_tree.commit('empty commit')
    self.build_tree(['tree/subtree/', 'tree/subtree/file1'])
    sub_tree = self.make_branch_and_tree('tree/subtree')
    sub_tree.add('file1', ids=b'file1-id')
    sub_tree.commit('added file1')
    return (base_tree, sub_tree)