import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def test_apply_delete(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('a', 'a\n')])
    tree.add('a')
    tree.commit('Add a')
    patch = parse_patch(b'--- a/a\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n'.splitlines(True))
    with AppliedPatches(tree, [patch]) as newtree:
        self.assertFalse(newtree.has_filename('a'))