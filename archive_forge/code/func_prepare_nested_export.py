import os
import tarfile
import zipfile
from breezy import osutils, tests
from breezy.errors import UnsupportedOperation
from breezy.export import export
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_tree import TestCaseWithTree
def prepare_nested_export(self, recurse_nested):
    tree = self.make_branch_and_tree('dir')
    self.build_tree(['dir/a'])
    tree.add('a')
    tree.commit('1')
    subtree = self.make_branch_and_tree('dir/subdir')
    self.build_tree(['dir/subdir/b'])
    subtree.add('b')
    subtree.commit('1a')
    try:
        tree.add_reference(subtree)
    except UnsupportedOperation:
        raise TestNotApplicable('format does not supported nested trees')
    tree.commit('2')
    export(tree, 'output', self.exporter, recurse_nested=recurse_nested)