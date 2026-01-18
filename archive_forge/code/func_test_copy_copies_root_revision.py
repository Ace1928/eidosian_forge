from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_copy_copies_root_revision(self):
    """Make sure the revision of the root gets copied."""
    inv = inventory.Inventory(root_id=b'someroot')
    inv.root.revision = b'therev'
    inv2 = inv.copy()
    self.assertEqual(b'someroot', inv2.root.file_id)
    self.assertEqual(b'therev', inv2.root.revision)