from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_file_detect_changes(self):
    left = inventory.InventoryFile(b'123', 'hello.c', ROOT_ID)
    left.text_sha1 = 123
    right = inventory.InventoryFile(b'123', 'hello.c', ROOT_ID)
    right.text_sha1 = 123
    self.assertEqual((False, False), left.detect_changes(right))
    self.assertEqual((False, False), right.detect_changes(left))
    left.executable = True
    self.assertEqual((False, True), left.detect_changes(right))
    self.assertEqual((False, True), right.detect_changes(left))
    right.text_sha1 = 321
    self.assertEqual((True, True), left.detect_changes(right))
    self.assertEqual((True, True), right.detect_changes(left))