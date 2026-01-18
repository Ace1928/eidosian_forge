from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def make_dir(self, inv, name, parent_id, revision):
    ie = inv.make_entry('directory', name, parent_id, name.encode('utf-8') + b'-id')
    ie.revision = revision
    inv.add(ie)