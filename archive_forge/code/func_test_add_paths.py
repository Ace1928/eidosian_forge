import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_paths(self):
    """Test smart-adding a list of paths."""
    paths = ('file1', 'file2')
    self.build_tree(paths)
    wt = self.make_branch_and_tree('.')
    wt.smart_add(paths)
    for path in paths:
        self.assertTrue(wt.is_versioned(path))