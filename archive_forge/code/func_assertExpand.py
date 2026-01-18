from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def assertExpand(self, all_ids, inv, file_ids):
    val_all_ids, val_children = inv._expand_fileids_to_parents_and_children(file_ids)
    self.assertEqual(set(all_ids), val_all_ids)
    entries = inv._getitems(val_all_ids)
    expected_children = {}
    for entry in entries:
        s = expected_children.setdefault(entry.parent_id, [])
        s.append(entry.file_id)
    val_children = {k: sorted(v) for k, v in val_children.items()}
    expected_children = {k: sorted(v) for k, v in expected_children.items()}
    self.assertEqual(expected_children, val_children)