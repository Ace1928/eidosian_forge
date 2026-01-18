from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def repository_text_keys(self):
    return {(b'a-file-id', b'broken-revision-1-2'): [(b'a-file-id', b'parent-1'), (b'a-file-id', b'parent-2')], (b'a-file-id', b'broken-revision-2-1'): [(b'a-file-id', b'parent-2'), (b'a-file-id', b'parent-1')], (b'a-file-id', b'parent-1'): [NULL_REVISION], (b'a-file-id', b'parent-2'): [NULL_REVISION]}