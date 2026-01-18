import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_dirstate_corrupt(self):
    error = dirstate.DirstateCorrupt('.bzr/checkout/dirstate', 'trailing garbage: "x"')
    self.assertEqualDiff('The dirstate file (.bzr/checkout/dirstate) appears to be corrupt: trailing garbage: "x"', str(error))