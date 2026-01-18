import os
from breezy import errors, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.tree import FileTimestampUnavailable
def test_get_renamed_time(self):
    """We should handle renamed files."""
    tree = self.make_basic_tree()
    tree.rename_one('one', 'two')
    st = os.lstat('tree/two')
    with tree.lock_read():
        mtime = tree.get_file_mtime('two')
        self.assertAlmostEqual(st.st_mtime, mtime)