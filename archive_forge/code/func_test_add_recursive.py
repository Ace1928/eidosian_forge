from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_add_recursive(self):
    parent = InventoryDirectory(b'src-id', 'src', b'tree-root')
    child = InventoryFile(b'hello-id', 'hello.c', b'src-id')
    parent.children[child.file_id] = child
    inv = inventory.Inventory(b'tree-root')
    inv.add(parent)
    self.assertEqual('src/hello.c', inv.id2path(b'hello-id'))