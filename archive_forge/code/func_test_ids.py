from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_ids(self):
    """Test detection of files within selected directories."""
    inv = inventory.Inventory(b'TREE_ROOT')
    inv.root.revision = b'revision'
    for args in [('src', 'directory', b'src-id'), ('doc', 'directory', b'doc-id'), ('src/hello.c', 'file'), ('src/bye.c', 'file', b'bye-id'), ('Makefile', 'file')]:
        ie = inv.add_path(*args)
        ie.revision = b'revision'
        if args[1] == 'file':
            ie.text_sha1 = osutils.sha_string(b'content\n')
            ie.text_size = len(b'content\n')
    inv = self.inv_to_test_inv(inv)
    self.assertEqual(inv.path2id('src'), b'src-id')
    self.assertEqual(inv.path2id('src/bye.c'), b'bye-id')