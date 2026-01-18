import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_tree_from_above_tree(self):
    """Test adding a tree from above the tree."""
    paths = ('original/', 'original/file1', 'original/file2')
    branch_paths = ('branch/', 'branch/original/', 'branch/original/file1', 'branch/original/file2')
    self.build_tree(branch_paths)
    wt = self.make_branch_and_tree('branch')
    wt.smart_add(('branch',))
    for path in paths:
        self.assertTrue(wt.is_versioned(path))