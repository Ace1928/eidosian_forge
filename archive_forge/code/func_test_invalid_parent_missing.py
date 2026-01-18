import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_invalid_parent_missing(self):
    state = self.assertBadDelta(active=[], basis=[], delta=[(None, 'path/path2', b'file-id')])
    state = self.assertBadDelta(active=[('path/', b'path-id')], basis=[], delta=[(None, 'path/path2', b'file-id')])
    state = self.assertBadDelta(active=[('path/', b'path-id'), ('path/path2', b'file-id')], basis=[], delta=[(None, 'path/path2', b'file-id')])