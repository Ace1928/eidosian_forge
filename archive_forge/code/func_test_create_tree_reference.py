from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_create_tree_reference(self):
    inv = inventory.Inventory(b'tree-root-123')
    inv.add(TreeReference(b'nested-id', 'nested', parent_id=b'tree-root-123', revision=b'rev', reference_revision=b'rev2'))