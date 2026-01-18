import os
from breezy import errors, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.tree import FileTimestampUnavailable
def test_get_renamed_in_subdir_time(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/d/', 'tree/d/a'])
    tree.add(['d', 'd/a'])
    rev_1 = tree.commit('1')
    tree.rename_one('d', 'e')
    st = os.lstat('tree/e/a')
    with tree.lock_read():
        mtime = tree.get_file_mtime('e/a')
        self.assertAlmostEqual(st.st_mtime, mtime)