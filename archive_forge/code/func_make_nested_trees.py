import os
from breezy import errors, tests, workingtree
from breezy.mutabletree import BadReferenceTarget
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def make_nested_trees(self):
    tree, sub_tree = self.make_trees()
    try:
        tree.add_reference(sub_tree)
    except errors.UnsupportedOperation:
        self._references_unsupported(tree)
    return (tree, sub_tree)