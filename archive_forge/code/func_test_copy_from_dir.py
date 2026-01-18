from io import StringIO
from .. import add, errors, tests
from ..bzr import inventory
def test_copy_from_dir(self):
    self.make_base_tree()
    new_tree = self.make_branch_and_tree('new')
    self.build_tree(['new/a', 'new/b', 'new/c', 'new/subdir/', 'new/subdir/b', 'new/subdir/d'])
    new_tree.set_root_id(self.base_tree.path2id(''))
    self.add_helper(self.base_tree, 'dir', new_tree, ['new'])
    self.assertEqual(self.base_tree.path2id('a'), new_tree.path2id('a'))
    self.assertEqual(self.base_tree.path2id('b'), new_tree.path2id('b'))
    self.assertEqual(self.base_tree.path2id('dir/subdir'), new_tree.path2id('subdir'))
    self.assertEqual(self.base_tree.path2id('dir/subdir/b'), new_tree.path2id('subdir/b'))
    c_id = new_tree.path2id('c')
    self.assertNotEqual(None, c_id)
    self.base_tree.lock_read()
    self.addCleanup(self.base_tree.unlock)
    self.assertRaises(errors.NoSuchId, self.base_tree.id2path, c_id)
    d_id = new_tree.path2id('subdir/d')
    self.assertNotEqual(None, d_id)
    self.assertRaises(errors.NoSuchId, self.base_tree.id2path, d_id)