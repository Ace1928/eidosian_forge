import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_custom_format(self):

    class BooDiffTree(DiffTree):

        def show_diff(self, specific_files, extra_trees=None):
            self.to_file.write('BOO!\n')
            return super().show_diff(specific_files, extra_trees)
    diff_format_registry.register('boo', BooDiffTree, 'Scary diff format')
    self.addCleanup(diff_format_registry.remove, 'boo')
    self.make_example_branch()
    self.build_tree_contents([('hello', b'hello world!\n')])
    output = self.run_bzr('diff --format=boo', retcode=1)
    self.assertTrue('BOO!' in output[0])
    output = self.run_bzr('diff -Fboo', retcode=1)
    self.assertTrue('BOO!' in output[0])