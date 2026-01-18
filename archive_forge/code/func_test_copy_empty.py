from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_copy_empty(self):
    """Make sure an empty inventory can be copied."""
    inv = inventory.Inventory(root_id=None)
    inv2 = inv.copy()
    self.assertIs(None, inv2.root)