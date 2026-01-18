import os
from breezy import ignores
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import run_script
def test_clean_tree_nested_bzrdir(self):
    wt1 = self.make_branch_and_tree('.')
    wt2 = self.make_branch_and_tree('foo')
    wt3 = self.make_branch_and_tree('bar')
    ignores.tree_ignores_add_patterns(wt1, ['./foo'])
    self.run_bzr(['clean-tree', '--unknown', '--force'])
    self.assertPathExists('foo')
    self.assertPathExists('bar')
    self.run_bzr(['clean-tree', '--ignored', '--force'])
    self.assertPathExists('foo')
    self.assertPathExists('bar')