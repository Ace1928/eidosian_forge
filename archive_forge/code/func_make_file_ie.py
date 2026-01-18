from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def make_file_ie(self, file_id=b'file-id', name='name', parent_id=None):
    ie_file = inventory.InventoryFile(file_id, name, parent_id)
    ie_file.revision = b'result'
    ie_file.text_size = 0
    ie_file.text_sha1 = b''
    return ie_file