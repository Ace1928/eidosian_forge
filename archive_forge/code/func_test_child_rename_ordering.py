import os
from breezy import tests
from breezy.bzr import inventory
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_child_rename_ordering(self):
    """Test the rename-parent, move child edge case.

        (A naive implementation may move the parent first, and then be
         unable to find the child.)
        """
    wt = self.make_branch_and_tree('.')
    root_id = wt.path2id('')
    self.build_tree(['dir/', 'dir/child', 'other/'])
    wt.add(['dir', 'dir/child', 'other'], ids=[b'dir-id', b'child-id', b'other-id'])
    wt.apply_inventory_delta([('dir', 'dir2', b'dir-id', inventory.InventoryDirectory(b'dir-id', 'dir2', root_id)), ('dir/child', 'other/child', b'child-id', inventory.InventoryFile(b'child-id', 'child', b'other-id'))])
    self.assertEqual('dir2', wt.id2path(b'dir-id'))
    self.assertEqual('other/child', wt.id2path(b'child-id'))