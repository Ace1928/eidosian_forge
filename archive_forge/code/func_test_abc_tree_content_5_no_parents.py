import shutil
from breezy import errors
from breezy.tests import TestNotApplicable, TestSkipped, features, per_tree
from breezy.tree import MissingNestedTree
def test_abc_tree_content_5_no_parents(self):
    tree = self.make_branch_and_tree('.')
    tree = self.get_tree_no_parents_abc_content_5(tree)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual([], tree.get_parent_ids())
    self.assertEqual([], tree.conflicts())
    self.assertEqual([], list(tree.unknowns()))
    self.assertEqual({'', 'd', 'b', 'b/c'}, set(tree.all_versioned_paths()))
    if tree.supports_file_ids:
        self.assertEqual([(p, tree.path2id(p)) for p in ['', 'b', 'd', 'b/c']], [(path, node.file_id) for path, node in tree.iter_entries_by_dir()])
    else:
        self.assertEqual(['', 'b', 'd', 'b/c'], [path for path, node in tree.iter_entries_by_dir()])
    self.assertEqualDiff(b'bar\n', tree.get_file_text('d'))
    self.assertFalse(tree.is_executable('b/c'))