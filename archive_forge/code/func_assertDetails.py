import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def assertDetails(self, expected, inv_entry):
    details = dirstate.DirState._inv_entry_to_details(inv_entry)
    self.assertEqual(expected, details)
    minikind, fingerprint, size, executable, tree_data = details
    self.assertIsInstance(minikind, bytes)
    self.assertIsInstance(fingerprint, bytes)
    self.assertIsInstance(tree_data, bytes)