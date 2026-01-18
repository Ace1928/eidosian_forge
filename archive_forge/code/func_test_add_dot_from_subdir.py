import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_dot_from_subdir(self):
    """Test adding . from a subdir of the tree."""
    paths = ('original/', 'original/file1', 'original/file2')
    self.build_tree(paths)
    wt = self.make_branch_and_tree('.')
    wt.smart_add(('.',))
    for path in paths:
        self.assertTrue(wt.is_versioned(path))