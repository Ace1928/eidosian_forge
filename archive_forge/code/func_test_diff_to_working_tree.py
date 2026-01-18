import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_to_working_tree(self):
    self.example_branch2()
    self.build_tree_contents([('branch1/file1', b'new line')])
    output = self.run_bzr('diff -r 1.. branch1', retcode=1)
    self.assertContainsRe(output[0], '\n\\-original line\n\\+new line\n')