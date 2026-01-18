from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_directory_has_text(self):
    dir = inventory.InventoryDirectory(b'123', 'hello.c', ROOT_ID)
    self.assertFalse(dir.has_text())