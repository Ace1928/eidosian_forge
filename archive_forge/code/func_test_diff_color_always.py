import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_color_always(self):
    from ... import colordiff
    from ...terminal import colorstring
    self.overrideAttr(colordiff, 'GLOBAL_COLORDIFFRC', None)
    self.example_branches()
    branch2_tree = workingtree.WorkingTree.open_containing('branch2')[0]
    self.build_tree_contents([('branch2/file', b'even newer content')])
    branch2_tree.commit(message='update file once more')
    out, err = self.run_bzr('diff --color=always -r revno:2:branch2..revno:1:branch1', retcode=1)
    self.assertEqual('', err)
    self.assertEqualDiff((colorstring(b"=== modified file 'file'\n", 'darkyellow') + colorstring(b'--- old/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n', 'darkred') + colorstring(b'+++ new/file\tYYYY-MM-DD HH:MM:SS +ZZZZ\n', 'darkblue') + colorstring(b'@@ -1 +1 @@\n', 'darkgreen') + colorstring(b'-new content\n', 'darkred') + colorstring(b'+contents of branch1/file\n', 'darkblue') + colorstring(b'\n', 'darkwhite')).decode(), subst_dates(out))