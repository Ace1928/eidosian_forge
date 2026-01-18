import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_dot_from_root(self):
    """Test adding . from the root of the tree."""
    paths = ('original/', 'original/file1', 'original/file2')
    self.build_tree(paths)
    wt = self.make_branch_and_tree('.')
    action = RecordingAddAction()
    wt.smart_add(('.',), action=action)
    for path in paths:
        self.assertTrue(wt.is_versioned(path))
    if wt.has_versioned_directories():
        self.assertEqual({(wt, 'original', 'directory'), (wt, 'original/file1', 'file'), (wt, 'original/file2', 'file')}, set(action.adds))
    else:
        self.assertEqual({(wt, 'original/file1', 'file'), (wt, 'original/file2', 'file')}, set(action.adds))