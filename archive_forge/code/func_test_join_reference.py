import os
from breezy import osutils, tests, workingtree
def test_join_reference(self):
    """Join can add a reference if --reference is supplied"""
    base_tree, sub_tree = self.make_trees()
    subtree_root_id = sub_tree.path2id('')
    self.run_bzr('join . --reference', working_dir='tree/subtree')
    sub_tree.lock_read()
    self.addCleanup(sub_tree.unlock)
    if sub_tree.supports_setting_file_ids():
        self.assertEqual(b'file1-id', sub_tree.path2id('file1'))
        self.assertEqual('file1', sub_tree.id2path(b'file1-id'))
        self.assertEqual(subtree_root_id, sub_tree.path2id(''))
        self.assertEqual('', sub_tree.id2path(subtree_root_id))
        self.assertEqual(sub_tree.path2id('file1'), base_tree.path2id('subtree/file1'))
    base_tree.lock_read()
    self.addCleanup(base_tree.unlock)
    self.assertEqual(['subtree'], list(base_tree.iter_references()))
    if base_tree.supports_setting_file_ids():
        self.assertEqual(b'file1-id', sub_tree.path2id('file1'))
        self.assertEqual('file1', sub_tree.id2path(b'file1-id'))
        self.assertEqual(subtree_root_id, base_tree.path2id('subtree'))
        self.assertEqual('subtree', base_tree.id2path(subtree_root_id))