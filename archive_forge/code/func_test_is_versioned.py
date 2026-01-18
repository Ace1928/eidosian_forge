from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_is_versioned(self):
    tree = self.make_branch_and_tree('tree')
    self.assertTrue(tree.is_versioned(''))
    self.assertFalse(tree.is_versioned('blah'))
    self.build_tree(['tree/dir/', 'tree/dir/file'])
    self.assertFalse(tree.is_versioned('dir'))
    self.assertFalse(tree.is_versioned('dir/'))
    tree.add(['dir', 'dir/file'])
    self.assertTrue(tree.is_versioned('dir'))
    self.assertTrue(tree.is_versioned('dir/'))