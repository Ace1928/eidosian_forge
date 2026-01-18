from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_link_has_text(self):
    link = inventory.InventoryLink(b'123', 'hello.c', ROOT_ID)
    self.assertFalse(link.has_text())