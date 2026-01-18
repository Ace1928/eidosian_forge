import os
from breezy import errors, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.tree import FileTimestampUnavailable
def test_after_commit(self):
    """Committing shouldn't change the mtime."""
    tree = self.make_basic_tree()
    st = os.lstat('tree/one')
    tree.commit('one')
    with tree.lock_read():
        mtime = tree.get_file_mtime('one')
        self.assertAlmostEqual(st.st_mtime, mtime)