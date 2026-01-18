import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def assertPackStat(self, expected, stat_value):
    """Check the packed and serialized form of a stat value."""
    self.assertEqual(expected, dirstate.pack_stat(stat_value))