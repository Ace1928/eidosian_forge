import os
from breezy.tests.per_tree import TestCaseWithTree
def test_get_file_with_stat_id_and_path(self):
    work_tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    work_tree.add(['foo'])
    tree = self._convert_tree(work_tree)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    file_obj, statvalue = tree.get_file_with_stat('foo')
    self.addCleanup(file_obj.close)
    if statvalue is not None:
        expected = os.lstat('foo')
        self.assertEqualStat(expected, statvalue)
    self.assertEqual([b'contents of foo\n'], file_obj.readlines())