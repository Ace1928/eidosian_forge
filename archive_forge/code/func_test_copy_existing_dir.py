from io import StringIO
from .. import add, errors, tests
from ..bzr import inventory
def test_copy_existing_dir(self):
    self.make_base_tree()
    new_tree = self.make_branch_and_tree('new')
    self.build_tree(['new/subby/', 'new/subby/a', 'new/subby/b'])
    subdir_file_id = self.base_tree.path2id('dir/subdir')
    new_tree.add(['subby'], ids=[subdir_file_id])
    self.add_helper(self.base_tree, '', new_tree, ['new'])
    self.assertEqual(self.base_tree.path2id('dir/subdir/b'), new_tree.path2id('subby/b'))
    a_id = new_tree.path2id('subby/a')
    self.assertNotEqual(None, a_id)
    self.base_tree.lock_read()
    self.addCleanup(self.base_tree.unlock)
    self.assertRaises(errors.NoSuchId, self.base_tree.id2path, a_id)