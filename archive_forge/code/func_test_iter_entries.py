from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_iter_entries(self):
    inv = self.prepare_inv_with_nested_dirs()
    self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('src', b'src-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id'), ('src/zz.c', b'zzc-id'), ('zz', b'zz-id')], [(path, ie.file_id) for path, ie in inv.iter_entries()])
    self.assertEqual([('bye.c', b'bye-id'), ('hello.c', b'hello-id'), ('sub', b'sub-id'), ('sub/a', b'a-id'), ('zz.c', b'zzc-id')], [(path, ie.file_id) for path, ie in inv.iter_entries(from_dir=b'src-id')])
    self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('src', b'src-id'), ('zz', b'zz-id')], [(path, ie.file_id) for path, ie in inv.iter_entries(recursive=False)])
    self.assertEqual([('bye.c', b'bye-id'), ('hello.c', b'hello-id'), ('sub', b'sub-id'), ('zz.c', b'zzc-id')], [(path, ie.file_id) for path, ie in inv.iter_entries(from_dir=b'src-id', recursive=False)])