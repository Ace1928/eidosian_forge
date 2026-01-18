import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_to_unversioned_non_ascii_dir(self):
    """Check error when moving to unversioned non-ascii directory"""
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', '§/'])
    tree.add(['a'])
    if tree.has_versioned_directories():
        e = self.assertRaises(errors.BzrMoveFailedError, tree.move, ['a'], '§')
        self.assertIsInstance(e.extra, errors.NotVersionedError)
        self.assertEqual(e.extra.path, '§')
    else:
        tree.move(['a'], '§')