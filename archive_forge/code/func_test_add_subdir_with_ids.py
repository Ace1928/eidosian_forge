from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_subdir_with_ids(self):
    tree = self.make_branch_and_tree('.')
    if not tree.supports_setting_file_ids():
        self.skipTest('tree does not support setting file ids')
    self.build_tree(['dir/', 'dir/subdir/', 'dir/subdir/foo'])
    tree.add(['dir'], ids=[b'dir-id'])
    tree.add(['dir/subdir'], ids=[b'subdir-id'])
    tree.add(['dir/subdir/foo'], ids=[b'foo-id'])
    root_id = tree.path2id('')
    self.assertTreeLayout([('', root_id), ('dir/', b'dir-id'), ('dir/subdir/', b'subdir-id'), ('dir/subdir/foo', b'foo-id')], tree)