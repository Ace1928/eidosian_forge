import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_invalid_changed_file(self):
    state = self.assertBadDelta(active=[('file', b'file-id')], basis=[], delta=[('file', 'file', b'file-id')])
    state = self.assertBadDelta(active=[('file', b'file-id')], basis=[('other-file', b'file-id')], delta=[('file', 'file', b'file-id')])