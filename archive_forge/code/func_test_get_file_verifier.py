from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_get_file_verifier(self):
    work_tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/file1', b'file content'), ('tree/file2', b'file content')])
    work_tree.add(['file1', 'file2'])
    tree = self._convert_tree(work_tree)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    kind, data = tree.get_file_verifier('file1')
    self.assertEqual(tree.get_file_verifier('file1'), tree.get_file_verifier('file2'))
    if kind == 'SHA1':
        expected = osutils.sha_string(b'file content')
        self.assertEqual(expected, data)