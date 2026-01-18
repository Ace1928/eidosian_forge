from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_file_has_text(self):
    file = inventory.InventoryFile(b'123', 'hello.c', ROOT_ID)
    self.assertTrue(file.has_text())