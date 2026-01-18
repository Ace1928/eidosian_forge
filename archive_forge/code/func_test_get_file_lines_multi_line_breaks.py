from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_get_file_lines_multi_line_breaks(self):
    work_tree = self.make_branch_and_tree('wt')
    self.build_tree_contents([('wt/foobar', b'a\rb\nc\r\nd')])
    work_tree.add('foobar')
    tree = self._convert_tree(work_tree)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual([b'a\rb\n', b'c\r\n', b'd'], tree.get_file_lines('foobar'))