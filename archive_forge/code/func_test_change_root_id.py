import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_change_root_id(self):
    state = self.assertUpdate(active=[('', b'root-id'), ('file', b'file-id')], basis=[('', b'root-id'), ('file', b'file-id')], target=[('', b'target-root-id'), ('file', b'file-id')])
    state = self.assertUpdate(active=[('', b'target-root-id'), ('file', b'file-id')], basis=[('', b'root-id'), ('file', b'file-id')], target=[('', b'target-root-id'), ('file', b'root-id')])
    state = self.assertUpdate(active=[('', b'active-root-id'), ('file', b'file-id')], basis=[('', b'root-id'), ('file', b'file-id')], target=[('', b'target-root-id'), ('file', b'root-id')])