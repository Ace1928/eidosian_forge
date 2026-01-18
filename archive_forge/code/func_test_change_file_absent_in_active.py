import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_change_file_absent_in_active(self):
    state = self.assertUpdate(active=[], basis=[('file', b'file-id')], target=[('file', b'file-id')])