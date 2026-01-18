import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_ignored_nested_paths(self):
    """Test smart-adding a list of paths which includes ignored ones."""
    wt = self.make_branch_and_tree('.')
    tree_shape = ('adir/', 'adir/CVS/', 'adir/CVS/afile', 'adir/CVS/afile2')
    add_paths = ('adir/CVS', 'adir/CVS/afile', 'adir')
    expected_paths = ('adir', 'adir/CVS', 'adir/CVS/afile', 'adir/CVS/afile2')
    self.build_tree(tree_shape)
    wt.smart_add(add_paths)
    for path in expected_paths:
        self.assertTrue(wt.is_versioned(path), 'No id added for %s' % path)